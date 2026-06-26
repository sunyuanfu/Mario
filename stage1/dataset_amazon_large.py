import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import dgl
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import roc_auc_score
class NodeClassificationDataset(object):
    def __init__(self, root: str,verbose: bool=True, device: str="cpu",bert_name: str = "bert-base-uncased",feat="clip",data_path="",save=False,trun=True):

        clip_model_name = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to("cuda")
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        graph_path = f"{root}/{self.name}_graph.dgl"
        img_dir = f"{root}/{self.name}"

        graph = torch.load(graph_path, weights_only=True)
        edges=graph["adjacency_matrix"]
        g = dgl.graph((edges[:, 0], edges[:, 1])) 
        self.graph=g
        batch_size=3000
        image=[]
        text=[]

        k=0
        self.label=[]
        file_path = os.path.join(root,f"{self.name}Images")
        textfeat=torch.empty(0)
        imgfeat=torch.empty(0)
        att=torch.empty(0)
        if save:
            for node_idx, node in graph["detail"].items():
              asin, desc, title, class_idx,reviews = node.values()
              img = Image.open(f"{img_dir}/{asin}.jpg")
              text.append("".join(desc)+title+"".join(reviews))
              image.append(img)
              self.label.append(class_idx)
              if len(image)==batch_size:
                with torch.no_grad():
                  with torch.cuda.amp.autocast():
                      inputs=self.processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True ).to("cuda")
                      outputs =self.model(**inputs)
          
                  if att.shape!=torch.empty(0).shape:
        
                      len_att = att.shape[1]
                      len_mask = inputs['attention_mask'].shape[1]
                      # Align both tensors to the same length
                      max_len = max(len_att, len_mask)

                      # Pad the shorter tensor with zeros to match the maximum length
                      if len_att < max_len:
                          padding = torch.zeros(att.size(0), max_len - len_att)
                          att = torch.cat([att, padding], dim=1)
                          padding = torch.zeros(att.size(0), max_len - len_att,512)
                          textfeat= torch.cat([textfeat, padding], dim=1)


                      if len_mask < max_len:
                          padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask).cuda()
                          inputs['attention_mask'] = torch.cat([inputs['attention_mask'], padding], dim=1)
                          padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask,512).cuda()
                          outputs.text_model_output.last_hidden_state= torch.cat([outputs.text_model_output.last_hidden_state, padding], dim=1)
                  print(att.shape)
                  print(inputs['attention_mask'].shape)
                  att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                  imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                  textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
                  print(imgfeat.shape)
                  print(textfeat.shape)
                  image=[]
                  text=[]
                  k+=batch_size
                  print(k)

            if image!=[]:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        inputs=self.processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True ).to("cuda")
                        outputs =self.model(**inputs)
            
                    if att.shape!=torch.empty(0).shape:
            
                        len_att = att.shape[1]
                        len_mask = inputs['attention_mask'].shape[1]
                        # Align both tensors to the same length
                        max_len = max(len_att, len_mask)

                        # Pad the shorter tensor with zeros to match the maximum length
                        if len_att < max_len:
                            padding = torch.zeros(att.size(0), max_len - len_att)
                            att = torch.cat([att, padding], dim=1)
                            padding = torch.zeros(att.size(0), max_len - len_att,512)
                            textfeat= torch.cat([textfeat, padding], dim=1)


                        if len_mask < max_len:
                            padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask).cuda()
                            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], padding], dim=1)
                            padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask,512).cuda()
                            outputs.text_model_output.last_hidden_state= torch.cat([outputs.text_model_output.last_hidden_state, padding], dim=1)
                    print(att.shape)
                    print(inputs['attention_mask'].shape)
                    att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                    imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                    textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
                    print(imgfeat.shape)
                    print(textfeat.shape)
                    image=[]
                    text=[]
    
                k+=batch_size
                print(k)       
            torch.save(imgfeat,os.path.join(root,"cimg_feat.pt"))
            print("img")
            print(imgfeat.dtype)

            torch.save(textfeat,os.path.join(root,"ctext_feat.pt"))
            print("text")
            print(textfeat.dtype)

            torch.save(att,os.path.join(root,"catt.pt"))
            print("att")
            print(att.dtype)

            torch.save(torch.tensor(self.label),os.path.join(root,"clabel.pt"))

            return 
  

        self.map=graph["subcategory"]

        self.graph.ndata['label']=torch.load(os.path.join(root,"clabel.pt"))
        self.label = self.graph.ndata['label']
        self.num_classes=max(self.label)+1
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")

            print(f'Device: {self.device}')


        self.num_nodes = self.graph.num_nodes()


        if not trun:
            node_ids=torch.load(os.path.join(root,"text_feat.pt"))
        else:
            node_ids=torch.load(os.path.join(root,"ctext_feat.pt"))
        self.graph.ndata['image_feat'] = torch.load(os.path.join(root,"cimg_feat.pt")).to(self.device)
        self.graph.ndata["text_feat"]=node_ids.to(self.device)
        self.graph.ndata['attention_mask']=torch.load(os.path.join(root,"catt.pt")).to(self.device)

        
        node_split_path = os.path.join(root, 'split.pt')
        self.node_split = self.split_graph(self.num_nodes,0.6,0.2)
        
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)

        train_mask[self.node_split[0]] = True
        val_mask[self.node_split[1]] = True
        test_mask[self.node_split[2]] = True
 
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata["index"]=torch.arange(self.num_nodes)
        self.graph.ndata["label"]=self.label
    def get_idx_split(self):
        return self.node_split

    def split_graph(self, nodes_num, train_ratio, val_ratio):
        np.random.seed(42)  # Ensure deterministic splits
        indices = np.random.permutation(nodes_num)  # Shuffle node indices

        # Compute train, validation, and test sizes
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        # Slice the node indices for each split
        train_ids = torch.tensor(indices[:train_size], dtype=torch.long)
        val_ids = torch.tensor(indices[train_size:train_size + val_size], dtype=torch.long)
        test_ids = torch.tensor(indices[train_size + val_size:], dtype=torch.long)

        # Return split sizes and torch tensor indices
        return train_ids, val_ids, test_ids


    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

# borrowed from OGB
class NodeClassificationEvaluator:
    def __init__(self, eval_metric: str):
        """
        Args:
            eval_metric (str): evaluation metric, can be "rocauc" or "acc".
        """
        self.num_tasks = 1
        self.eval_metric = eval_metric


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
