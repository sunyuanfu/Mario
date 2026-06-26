"""Mario Stage-2 LLM model definition."""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Sequence, Tuple

import dgl
import torch


def _bitsandbytes_abstract_compat(*args, **kwargs):
    def decorator(function):
        return function

    return decorator


torch.library.impl_abstract = _bitsandbytes_abstract_compat
if hasattr(torch.library, "register_fake"):
    torch.library.register_fake = _bitsandbytes_abstract_compat

import torch.nn.functional as F
import peft.import_utils as peft_import_utils
import peft.tuners.lora.model as peft_lora_model
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM

from .constants import IGNORE_INDEX, TEMPLATES
from .data import build_adjacency


peft_import_utils.is_bnb_available = lambda: False
peft_import_utils.is_bnb_4bit_available = lambda: False
peft_lora_model.is_bnb_available = lambda: False
peft_lora_model.is_bnb_4bit_available = lambda: False


class MarioLLM(nn.Module):
    """Stage-2 LoRA-tuned LLM with MAPR-style routing objective."""

    def __init__(
        self,
        args: argparse.Namespace,
        graph: dgl.DGLGraph,
        texts: Sequence[str],
        labels: torch.Tensor,
        label_names: Sequence[str],
        train_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.args = args
        self.graph = graph
        self.texts = list(texts)
        self.labels = labels.cpu()
        self.label_names = list(label_names)
        self.features = F.normalize(features.detach().cpu(), dim=-1).detach()
        self.train_nodes = set(train_idx.cpu().tolist())
        self.adjacency = build_adjacency(graph)
        self.temperature = args.router_temperature
        self.kl_weight = args.kl_weight
        self.max_text_tokens = args.max_text_tokens
        self.max_new_tokens = args.max_new_tokens
        self.top_k = args.top_k
        self.last_info: Dict[str, float] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path,
            local_files_only=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.llm = LlamaForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(args.device)
        self.llm.config.use_cache = False
        checkpoint_kwargs = {"use_reentrant": False}
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpoint_kwargs
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.hidden_size = int(self.llm.get_input_embeddings().embedding_dim)
        feature_dim = int(self.features.size(-1))
        # Shared projector P used to map Stage-1 features into the LLM embedding space.
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(args.device)
        router_hidden = 2048
        self.router = nn.Sequential(
            nn.Linear(feature_dim * 4 + 1, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, router_hidden // 2),
            nn.GELU(),
            nn.Linear(router_hidden // 2, 3),
        ).to(args.device)

    @property
    def device(self) -> torch.device:
        return next(self.router.parameters()).device

    def trainable_parameters(self) -> Tuple[int, int]:
        trainable = 0
        total = 0
        for parameter in self.parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
        return trainable, total

    def token_embedding(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            return torch.empty(0, self.hidden_size, device=self.device, dtype=self.llm.dtype)
        token_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        return self.llm.get_input_embeddings()(token_ids)

    def limited_raw_text_embedding(self, node_id: int) -> torch.Tensor:
        ids = self.tokenizer(self.texts[node_id], add_special_tokens=False).input_ids[
            : self.max_text_tokens
        ]
        if not ids:
            return self.token_embedding(" ")
        token_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        return self.llm.get_input_embeddings()(token_ids)

    def feature_token(self, node_id: int, modality: str) -> torch.Tensor:
        feature = self.features[node_id].to(self.device)
        if modality == "text":
            token = self.projector(feature[0].unsqueeze(0))
        elif modality == "image":
            token = self.projector(feature[1].unsqueeze(0))
        else:
            text_token = self.projector(feature[0].unsqueeze(0))
            image_token = self.projector(feature[1].unsqueeze(0))
            token = torch.cat([text_token, image_token], dim=0)
        return token.to(dtype=self.llm.dtype)

    def topk_neighbors(self, center: int, candidates: Sequence[int], modality: str) -> List[int]:
        valid = [node for node in candidates if node in self.train_nodes and node != center]
        if not valid:
            return []
        center_feature = torch.cat([self.features[center, 0], self.features[center, 1]], dim=-1)
        neighbor_feature = torch.cat([self.features[valid, 0], self.features[valid, 1]], dim=-1)
        scores = F.cosine_similarity(neighbor_feature, center_feature.unsqueeze(0), dim=-1)
        count = min(self.top_k, len(valid))
        order = torch.topk(scores, k=count).indices.tolist()
        return [valid[index] for index in order]

    def context_nodes(self, center: int, modality: str) -> Tuple[List[int], List[int]]:
        one_candidates = [node for node in self.adjacency[center] if node != center]
        one_hop = self.topk_neighbors(center, one_candidates, modality)
        two_candidates = set()
        for neighbor in one_candidates:
            two_candidates.update(self.adjacency[neighbor])
        two_candidates.difference_update(one_candidates)
        two_candidates.discard(center)
        two_hop = self.topk_neighbors(center, sorted(two_candidates), modality)
        return one_hop, two_hop

    def pooled_context(self, nodes: Sequence[int]) -> torch.Tensor:
        feature_dim = int(self.features.size(-1))
        if not nodes:
            return torch.zeros(feature_dim, device=self.device)
        values = self.features[list(nodes)].to(self.device)  # [N, 2, d]
        pooled = (values[:, 0] + values[:, 1]) / 2.0
        return pooled.mean(dim=0)

    def router_logits(self, center: int, one_hop: Sequence[int], two_hop: Sequence[int]) -> torch.Tensor:
        center_feature = self.features[center].to(self.device)
        center_text = center_feature[0]
        center_image = center_feature[1]
        one = self.pooled_context(one_hop)
        two = self.pooled_context(two_hop)
        degree = torch.tensor([math.log1p(len(self.adjacency[center]))], device=self.device)
        router_input = torch.cat([center_text, center_image, one, two, degree], dim=-1).unsqueeze(0)
        return self.router(router_input)

    def template_intro(self, center: int, modality: str) -> str:
        # The instruction is concise (e.g., "Predict the node category.").
        # We keep the instruction fixed across templates and vary only the modality-specific signals.
        return "Predict the node category.\nAnchor raw text: "

    def append_labeled_neighbors(
        self,
        parts: List[torch.Tensor],
        nodes: Sequence[int],
        modality: str,
        hop: int,
    ) -> None:
        parts.append(self.token_embedding("\nHop {} labeled neighbors: ".format(hop)))
        if not nodes:
            parts.append(self.token_embedding("none. "))
            return
        for position, node in enumerate(nodes, start=1):
            label = self.label_names[int(self.labels[node].item())]
            parts.append(self.token_embedding("N{} feature ".format(position)))
            if modality == "text":
                parts.append(self.feature_token(node, "text"))
            elif modality == "image":
                parts.append(self.feature_token(node, "image"))
            else:
                parts.append(self.feature_token(node, "mm"))
            parts.append(self.token_embedding(" category: {}; ".format(label)))

    def build_template(
        self,
        center: int,
        modality: str,
        include_label: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        one_hop, two_hop = self.context_nodes(center, modality)
        parts: List[torch.Tensor] = []
        parts.append(self.token_embedding(self.template_intro(center, modality)))
        parts.append(self.limited_raw_text_embedding(center))
        parts.append(self.token_embedding("\nAnchor {} signal: ".format(modality)))
        if modality == "text":
            parts.append(self.feature_token(center, "text"))
        elif modality == "image":
            parts.append(self.feature_token(center, "image"))
        else:
            parts.append(self.feature_token(center, "mm"))
        self.append_labeled_neighbors(parts, one_hop, modality, 1)
        self.append_labeled_neighbors(parts, two_hop, modality, 2)
        parts.append(self.token_embedding("\nAnswer:"))
        prefix = torch.cat(parts, dim=0)
        if include_label:
            label_name = self.label_names[int(self.labels[center].item())]
            target_ids = self.tokenizer(" " + label_name, add_special_tokens=False).input_ids + [
                self.tokenizer.eos_token_id
            ]
            target_tensor = torch.tensor(target_ids, device=self.device, dtype=torch.long)
            target_embeds = self.llm.get_input_embeddings()(target_tensor)
            bos = self.llm.get_input_embeddings()(
                torch.tensor([self.tokenizer.bos_token_id], device=self.device)
            )
            inputs = torch.cat([bos, prefix, target_embeds], dim=0)
            labels = torch.full(
                (inputs.size(0),),
                IGNORE_INDEX,
                dtype=torch.long,
                device=self.device,
            )
            labels[-len(target_ids) :] = target_tensor
        else:
            bos = self.llm.get_input_embeddings()(
                torch.tensor([self.tokenizer.bos_token_id], device=self.device)
            )
            inputs = torch.cat([bos, prefix], dim=0)
            labels = torch.full(
                (inputs.size(0),),
                IGNORE_INDEX,
                dtype=torch.long,
                device=self.device,
            )
        return inputs.to(dtype=self.llm.dtype), labels, one_hop, two_hop

    def pad_templates(
        self, templates: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(item[0].size(0) for item in templates)
        pad_id = self.tokenizer.pad_token_id
        pad_embed = self.llm.get_input_embeddings()(
            torch.tensor([pad_id], device=self.device)
        ).to(dtype=self.llm.dtype)
        input_rows = []
        label_rows = []
        mask_rows = []
        for inputs, labels in templates:
            pad_len = max_len - inputs.size(0)
            if pad_len > 0:
                inputs = torch.cat([pad_embed.repeat(pad_len, 1), inputs], dim=0)
                labels = torch.cat(
                    [
                        torch.full(
                            (pad_len,),
                            IGNORE_INDEX,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        labels,
                    ],
                    dim=0,
                )
                mask = torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.long, device=self.device),
                        torch.ones(
                            inputs.size(0) - pad_len,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                mask = torch.ones(inputs.size(0), dtype=torch.long, device=self.device)
            input_rows.append(inputs)
            label_rows.append(labels)
            mask_rows.append(mask)
        return (
            torch.stack(input_rows, dim=0),
            torch.stack(label_rows, dim=0),
            torch.stack(mask_rows, dim=0),
        )

    def per_template_losses(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size(1) != labels.size(1):
            labels = labels[:, -logits.size(1) :]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.size())
        active = shift_labels.ne(IGNORE_INDEX).float()
        return (flat_loss * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(centers):
            centers_list = [int(node) for node in centers.detach().cpu().view(-1).tolist()]
        else:
            centers_list = [int(node) for node in centers]
        loss, info = self.forward_nodes(centers_list)
        self.last_info = info
        return loss

    def forward_node(self, center: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.forward_nodes([center])

    def forward_nodes(self, centers: Sequence[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        all_templates = []
        router_contexts = []
        for center in centers:
            built = [self.build_template(int(center), template, True) for template in TEMPLATES]
            all_templates.extend((item[0], item[1]) for item in built)
            router_contexts.append((int(center), built[2][2], built[2][3]))
        inputs, labels, mask = self.pad_templates(all_templates)
        target_lengths = labels.ne(IGNORE_INDEX).sum(dim=1)
        logits_to_keep = int(target_lengths.max().item()) + 1
        outputs = self.llm(
            inputs_embeds=inputs,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
        losses = self.per_template_losses(outputs.logits, labels).view(
            len(centers), len(TEMPLATES)
        )
        router_logits = torch.cat(
            [self.router_logits(center, one_hop, two_hop) for center, one_hop, two_hop in router_contexts],
            dim=0,
        )
        probs = F.softmax(router_logits, dim=-1)
        posterior = F.softmax(-losses.detach() / self.temperature, dim=-1)
        weighted = (posterior * losses).sum(dim=1)
        kl = (
            posterior
            * (torch.log(posterior.clamp_min(1e-8)) - torch.log(probs.clamp_min(1e-8)))
        ).sum(dim=1)
        per_node = weighted + self.kl_weight * kl
        loss = per_node.mean()
        info = {
            "loss": float(loss.detach().cpu()),
            "text_loss": float(losses[:, 0].mean().detach().cpu()),
            "image_loss": float(losses[:, 1].mean().detach().cpu()),
            "mm_loss": float(losses[:, 2].mean().detach().cpu()),
            "router_text": float(probs[:, 0].mean().detach().cpu()),
            "router_image": float(probs[:, 1].mean().detach().cpu()),
            "router_mm": float(probs[:, 2].mean().detach().cpu()),
        }
        return loss, info

    @torch.no_grad()
    def validation_loss(self, center: int) -> float:
        was_training = self.training
        self.eval()
        loss, _ = self.forward_node(center)
        if was_training:
            self.train()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def _router_probs(self, center: int) -> Dict[str, float]:
        one_hop, two_hop = self.context_nodes(center, "mm")
        probs = F.softmax(self.router_logits(center, one_hop, two_hop), dim=-1).squeeze(0)
        return {template: float(probs[idx].detach().cpu()) for idx, template in enumerate(TEMPLATES)}

    @torch.no_grad()
    def _generate_with_template(self, center: int, template: str) -> str:
        inputs, _, _, _ = self.build_template(center, template, False)
        attention = torch.ones(1, inputs.size(0), dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            0, inputs.size(0), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        outputs = self.llm.generate(
            inputs_embeds=inputs.unsqueeze(0),
            attention_mask=attention,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prefix_len = int(inputs.size(0))
        if outputs.size(1) > prefix_len:
            gen_ids = outputs[0][prefix_len:]
        else:
            gen_ids = outputs[0]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def predict_node(self, center: int, policy: str = "router") -> str:
        self.eval()
        if policy in TEMPLATES:
            template = policy
        else:
            router_probs = self._router_probs(center)
            template = max(router_probs.items(), key=lambda item: item[1])[0]
        return self._generate_with_template(center, template)

    @torch.no_grad()
    def predict_node_all_templates(self, center: int) -> Tuple[Dict[str, str], Dict[str, float]]:
        self.eval()
        texts = {
            template: self._generate_with_template(center, template)
            for template in TEMPLATES
        }
        router_probs = self._router_probs(center)
        return texts, router_probs
