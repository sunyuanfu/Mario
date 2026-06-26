import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jsonlines
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import dgl
from transformers import CLIPProcessor, CLIPModel
import psutil
import os
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizerFast,BertModel,ViTFeatureExtractor, ViTForImageClassification,ViTModel
class NodeClassificationDataset(object):
    def __init__(self, root: str,verbose: bool=True, device: str="cpu",bert_name: str = "bert-base-uncased",feat="clip",data_path="",save=False,trun=True,use_large_features=False):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            verbose (bool): whether to print the information.
            device (str): device to use.
            use_large_features (bool): whether to use large features from ImageFeature/TextFeature folders
        """
        root = os.path.normpath(root)
        # Only load CLIP model if save=True (for feature extraction)
        if save:
            clip_model_name = "openai/clip-vit-base-patch16"
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.model = CLIPModel.from_pretrained(clip_model_name).to("cuda")
        transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor()
        ])
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        df = pd.read_csv(f"{root}/{self.name}.csv")
        gpth=f"{root}/{self.name}Graph.pt"
        graph = dgl.load_graphs(gpth)[0][0]
        self.label = labels = graph.ndata['label']
        self.num_classes=max(self.label)+1
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Device: {self.device}')

        self.num_nodes = graph.num_nodes()
        batch_size=3000
        image=[]
        text=[]

        k=0
        
        file_path = os.path.join(root,f"{self.name}Images")
        textfeat=torch.empty(0)
        imgfeat=torch.empty(0)
        att=torch.empty(0)
        
        if save:
            for stu in range(df.shape[0]):
                img_path = os.path.join(file_path, f"{stu}.jpg")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                    except Exception as e:
                        img = Image.new("RGB", (100, 100), "white") 
                else:
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                if img.mode == 'L':
                    img = Image.merge("RGB", (img, img, img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                text.append(df.loc[stu,"caption"])
                image.append(img)

                if len(image)==batch_size:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            inputs=self.processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True ).to("cuda")
                            outputs =self.model(**inputs)
                    
                        if att.shape!=torch.empty(0).shape:
                            len_att = att.shape[1]
                            len_mask = inputs['attention_mask'].shape[1]
                            max_len = max(len_att, len_mask)
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
                        
                        att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                        imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                        textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
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
                        max_len = max(len_att, len_mask)
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
                    
                    att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                    imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                    textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
                    image=[]
                    text=[]
    
                k+=batch_size
                print(k)
                    
            torch.save(imgfeat,os.path.join(root,"cimg_feat.pt"))
            torch.save(textfeat,os.path.join(root,"ctext_feat.pt"))
            torch.save(att,os.path.join(root,"catt.pt"))
            return 

        # 使用大图特征文件（4096维）
        if use_large_features:
            print("Loading large features from ImageFeature and TextFeature folders...")
            img_feat_path = os.path.join(root, "ImageFeature", f"{self.name}_Llama-3.2-11B-Vision-Instruct_visual.npy")
            text_feat_path = os.path.join(root, "TextFeature", f"{self.name}_Llama_3.2_11B_Vision_Instruct_512_mean.npy")
            
            img_feat_np = np.load(img_feat_path)
            text_feat_np = np.load(text_feat_path)
            
            img_feat = torch.from_numpy(img_feat_np).float()
            text_feat = torch.from_numpy(text_feat_np).float()
            
            # 将特征移动到目标设备
            img_feat = img_feat.to(self.device)
            text_feat = text_feat.to(self.device)
            
            # 使用DGL的方法创建图并移动到设备
            src, dst = graph.edges()
            self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
            
            # 先添加特征到CPU图，然后整体移动到目标设备
            self.graph.ndata['image_feat'] = img_feat.cpu().unsqueeze(1)
            self.graph.ndata["text_feat"] = text_feat.cpu().unsqueeze(1)
            self.graph.ndata['attention_mask'] = torch.ones(self.num_nodes, 1, dtype=torch.long)
            
            print(f"Loaded large features:")
            print(f"  Image features shape: {img_feat.shape}")
            print(f"  Text features shape: {text_feat.shape}")
            print(f"  Graph nodes: {self.num_nodes}")
            
        # 使用小特征文件（512维）
        else:
            if not trun:
                node_ids=torch.load(os.path.join(root,"text_feat.pt"))
            else:
                node_ids=torch.load(os.path.join(root,"ctext_feat.pt"))

            src, dst = graph.edges()
            # 先在CPU上创建图
            self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
            # 先添加特征到CPU图
            self.graph.ndata['image_feat'] = torch.load(os.path.join(root,"cimg_feat.pt"))
            self.graph.ndata["text_feat"] = node_ids
            self.graph.ndata['attention_mask'] = torch.load(os.path.join(root,"catt.pt"))

        
        node_split_path = os.path.join(root, 'split.pt')
        self.node_split = self.split_graph(self.num_nodes,0.6,0.2)
        
        # 先在CPU上创建mask
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[self.node_split[0]] = True
        val_mask[self.node_split[1]] = True
        test_mask[self.node_split[2]] = True
 
        # 添加mask到图
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata["index"] = torch.arange(self.num_nodes)
        self.graph.ndata["label"] = self.label
        
        # 如果目标设备是CUDA，使用CUDA张量重新创建图
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            src, dst = self.graph.edges()
            src = src.cuda()
            dst = dst.cuda()
            new_graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
            # 移动所有特征
            for key in self.graph.ndata:
                new_graph.ndata[key] = self.graph.ndata[key].cuda()
            self.graph = new_graph
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
