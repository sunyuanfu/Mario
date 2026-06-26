import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnlrv3.convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert
from src.models.tnlrv3.modeling import TuringNLRv3PreTrainedModel, logger, BertSelfAttention, BertLayer, WEIGHTS_NAME, \
    BertEmbeddings, relative_position_bucket
from src.utils import roc_auc_score, mrr_score, ndcg_score
from dgl.nn import GATv2Conv

class GraphTuringNLRPreTrainedModel(TuringNLRv3PreTrainedModel):
    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=None,
            replace_prefix=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'tnlrv3')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME),
                                            map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("TNLRv3 does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        # initialize new rel_pos weight
        _k = 'bert.rel_pos_bias.weight'
        if _k in state_dict and state_dict[_k].shape[1] != (config.rel_pos_bins + 2):
            logger.info(
                f"rel_pos_bias.weight.shape[1]:{state_dict[_k].shape[1]} != config.bus_num+config.rel_pos_bins:{config.rel_pos_bins + 2}")
            old_rel_pos_bias = state_dict[_k]
            new_rel_pos_bias = torch.cat(
                [old_rel_pos_bias, old_rel_pos_bias[:, -1:].expand(old_rel_pos_bias.size(0), 2)], -1)
            new_rel_pos_bias = nn.Parameter(data=new_rel_pos_bias, requires_grad=True)
            state_dict[_k] = new_rel_pos_bias.data
            del new_rel_pos_bias

        if replace_prefix is not None:
            new_state_dict = {}
            for key in state_dict:
                if key.startswith(replace_prefix):
                    new_state_dict[key[len(replace_prefix):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            kwargs["state_dict"] = new_state_dict
            del state_dict

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask=None, rel_pos=None):
        query = self.query(hidden_states[:, :1])  # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        station_embed = self.multi_head_attention(query=query,
                                                  key=key,
                                                  value=value,
                                                  attention_mask=attention_mask,
                                                  rel_pos=rel_pos)[0]  # B 1 D
        station_embed = station_embed.squeeze(1)

        return station_embed


class GraphBertEncoder(nn.Module):
    def __init__(self, config,arg):
        super(GraphBertEncoder, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.arg=arg
        num_heads = 2
        num_out_heads = 1
        activation = F.elu
        feat_drop = 0
        attn_drop = 0
        negative_slope = 0.2
        residual = True
        heads = ([num_heads] * (arg.num_layers - 1)) + [num_out_heads]
        self.graph_attention = GATv2(arg.hidden_dim,arg.hidden_dim,arg.num_layers,heads,activation, feat_drop, attn_drop,
                        negative_slope, residual)
        self.gnn=True
    def forward(self,
                hidden_states,
                attention_mask,
                node_mask=None,
                node_rel_pos=None,
                rel_pos=None,
                block=None):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape


        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                # B SN L D
                if self.gnn:
                    cls_emb = hidden_states[:, 0, :].clone()  # B SN D
                    station_emb = self.graph_attention(block,cls_emb)  # B D

                    # update the station in the query/key

                    hidden_states[:station_emb.shape[0], 0, :] = station_emb
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, rel_pos=rel_pos)

            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:, :, :, 0] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask, rel_pos=rel_pos)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphFormers(TuringNLRv3PreTrainedModel):
    def __init__(self, config,arg):
        super(GraphFormers, self).__init__(config=config)
        self.config = config
        self.encoder = GraphBertEncoder(config=config,arg=arg)
        #####################################
        self.config.rel_pos_bins=32
        self.config.max_rel_pos = 128
        ####################################
        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins + 2,
                                          config.num_attention_heads,
                                          bias=False)
        else:
            self.rel_pos_bias = None
        
        self.input_proj = nn.Linear(4096, config.hidden_size) if hasattr(arg, 'use_large_features') and arg.use_large_features else None

    def forward(self,
                block,
                input_feat,
                attention_mask):
        all_nodes_num, seq_length,_ = input_feat.shape

        if self.input_proj is not None:
            input_feat = self.input_proj(input_feat)
        embedding_output= input_feat
        seq_length-=1
        device = input_feat.device if input_feat is not None else input_feat.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).repeat(input_feat.shape[0],1)
        batch_size, subgraph_node_num=1,input_feat.shape[0]
        #attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L
        #attention_mask[:, 0] = 1.0  # only use the station for main nodes
  
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        if self.config.rel_pos_bins > 0:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.config.rel_pos_bins,
                                               max_distance=self.config.max_rel_pos)

            # rel_pos: (N,L,L) -> (N,1+L,L)
            temp_pos = torch.zeros(all_nodes_num, 1, seq_length, dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([temp_pos, rel_pos], dim=1)
            # rel_pos: (N,1+L,L) -> (N,1+L,1+L)
            station_relpos = torch.full((all_nodes_num, seq_length + 1, 1), self.config.rel_pos_bins,
                                        dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([station_relpos, rel_pos], dim=-1)

            # node_rel_pos:(B:batch_size, Head_num, neighbor_num+1)
            node_pos = self.config.rel_pos_bins + 1
            node_rel_pos = torch.full((batch_size, subgraph_node_num), node_pos, dtype=rel_pos.dtype,
                                      device=rel_pos.device)
            node_rel_pos[:, 0] = 0
            node_rel_pos = F.one_hot(node_rel_pos,
                                     num_classes=self.config.rel_pos_bins + 2).type_as(
                embedding_output)
            node_rel_pos = self.rel_pos_bias(node_rel_pos).permute(0, 2, 1)  # B head_num, neighbor_num
            node_rel_pos = node_rel_pos.unsqueeze(2)  # B head_num 1 neighbor_num

            # rel_pos: (N,Head_num,1+L,1+L)
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins + 2).type_as(
                embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        else:
            node_rel_pos = None
            rel_pos = None

        # Add station_placeholder
   # N 1+L D

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            node_rel_pos=node_rel_pos,
            rel_pos=rel_pos,
            block=block)

        return encoder_outputs


class GraphFormersForNeighborPredict(GraphTuringNLRPreTrainedModel):
    def __init__(self, config,arg):
        super().__init__(config)
        self.bert = GraphFormers(config,arg)
        self.image_bert = GraphFormers(config, arg)
        self.init_weights()
        self.arg=arg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def infer(self, block,feat,attention_mask):
        # B, N, L = input_ids_node_and_neighbors_batch.shape
        # D = self.config.hidden_size
        # input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        # attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)

        hidden_states = self.bert(block,feat,attention_mask)
        last_hidden_states = hidden_states[0]
        cls_embeddings = last_hidden_states[:, 0]

        node_embeddings = cls_embeddings


        return node_embeddings
    def image_infer(self,  block,feat):
        # B, N, L = input_ids_node_and_neighbors_batch.shape
        # D = self.config.hidden_size
        # input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        # attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)
        attention_mask=torch.zeros((feat.shape[0],feat.shape[1])).to(feat.device)
        hidden_states = self.image_bert(block,feat,attention_mask)
        last_hidden_states = hidden_states[0]
        cls_embeddings = last_hidden_states[:, 0]
        node_embeddings = cls_embeddings
        return node_embeddings

    def test(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
             mask_query_and_neighbors_batch, \
             input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, mask_key_and_neighbors_batch,
             **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                      mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    mask_key_and_neighbors_batch)
        scores = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, block,text_feat,attention_mask,image_feat,num_node=None,out=True):

        text_embeddings = self.infer(block,text_feat,attention_mask)
        image_embeddings = self.image_infer(block,image_feat)
        
        if out:
            return text_embeddings,image_embeddings
        text_embedding, image_embedding = text_embeddings[:num_node,], image_embeddings[:num_node,]
        epsilon = 1e-8
            
        image_embedding = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + epsilon)
        text_embedding = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + epsilon)
        logits_per_image = self.logit_scale.exp() * image_embedding @ text_embedding.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image,logits_per_text

class GATv2(nn.Module):
    def __init__(self, in_size, hid_size, num_layers, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads
        self.hid_size = hid_size
        self.layer_norms = torch.nn.ModuleList()
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(in_size, hid_size, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation,
                      bias=False, share_weights=True)
        )
        # hidden layers
        for l in range(num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layer_norms.append(nn.LayerNorm(hid_size * heads[l]))
            self.gatv2_layers.append(
                GATv2Conv(hid_size * heads[l], hid_size, heads[l + 1], feat_drop, attn_drop, negative_slope, residual,
                          self.activation, bias=False, share_weights=True)
            )
        # output projection
        self.predictor = nn.Sequential(
            nn.Linear(hid_size * heads[-1], hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gatv2_layers, blocks)):
            h = layer(block, h).flatten(1)
            if l != len(self.gatv2_layers) - 1:
                h = self.layer_norms[l](h)
                h = F.relu(h)
        return h

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.gatv2_layers):
            y = torch.empty(g.num_nodes(), self.hid_size * self.heads[l], device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
