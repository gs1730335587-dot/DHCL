import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
# from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

# model_hyper.py 顶部添加

# ===== 新增：导入对比学习模块 =====
from contrastive_loss import DualGraphContrastiveLearning, SimpleContrastiveLearning


def topk(x, ratio, batch, min_score=None, tol=1e-7):

    if min_score is not None:

        perm = torch.nonzero(x >= min_score, as_tuple=False).view(-1)
    else:

        if ratio is not None:
            num_nodes = x.size(0)
            k = max(1, int(ratio * num_nodes))
        else:
            k = x.size(0)

        perm = torch.topk(x, k, dim=0)[1]

    return perm


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):

        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)


class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4,
                use_contrastive=False, contrastive_temp=0.07, use_simple_contrast=False):  # 新增参数
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K


        self.adho_att = nn.Linear(nhidden, 1)
        self.adho_mlp = nn.Sequential(
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, 1)
        )

        self.tau_start = 5.0
        self.tau_end = 1.0
        self.soft_alpha = 0.4
        self.sparse_theta = 0.25
        self.sparse_lambda = 5e-5


        self.use_contrastive = use_contrastive
        self.use_simple_contrast = use_simple_contrast
        if self.use_contrastive:
            if self.use_simple_contrast:

                self.contrastive_learner = SimpleContrastiveLearning(
                    hidden_dim=nhidden,
                    temperature=contrastive_temp
                )
            else:
                self.contrastive_learner = DualGraphContrastiveLearning(
                    hidden_dim=nhidden,
                    temperature=contrastive_temp
                )
        
        self.diffusion = None
        self.freq_diffuser = None
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))

    def forward(self, a, v, l, dia_len, qmask, epoch, labels=None):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  


        adho_sparse_loss = None
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max().item()) + 1
            node_idx = hyperedge_index[0]                # [E_inc]
            edge_idx = hyperedge_index[1]                # [E_inc]


            node_logits = self.adho_att(x1).squeeze(-1)  # [N]

            inc_logits = node_logits[node_idx]           # [E_inc]

            alpha_ij = softmax(inc_logits, edge_idx)     # [E_inc]


            h_i = x1[node_idx]                           # [E_inc, nhidden]
            weighted_h = alpha_ij.unsqueeze(-1) * h_i    # [E_inc, nhidden]
            z_j = scatter_add(weighted_h, edge_idx, dim=0, dim_size=num_edges)  # [num_edges, nhidden]


            s_j = torch.sigmoid(self.adho_mlp(z_j))      # [num_edges, 1]


            with torch.no_grad():

                cur_epoch = 0 if epoch is None else max(int(epoch), 0)
                tau = max(self.tau_end, self.tau_start * (0.99 ** cur_epoch))
            eps = 1e-8
            logits = torch.log(s_j + eps) - torch.log(1 - s_j + eps)  # Bernoulli 对数几率
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
            p_j = torch.sigmoid((logits + gumbel_noise) / tau)        # [num_edges,1]


            w_j = self.soft_alpha + (1.0 - self.soft_alpha) * p_j     # [num_edges,1]
            w_j = w_j.view(-1)                                        # [num_edges]


            base_weight = self.hyperedge_weight[0:num_edges]
            weight = base_weight * w_j


            mean_p = p_j.mean()
            adho_sparse_loss = self.sparse_lambda * F.relu(self.sparse_theta - mean_p)
        else:
            weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   

        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out, gnn_edge_index)


        contrastive_loss = None
        contrastive_dict = {}
        
        if self.use_contrastive and self.training:
            N = sum(dia_len)
            hyper_out_for_cl = out[0:N, :]
            gnn_out_for_cl = gnn_out[0:N, :]
            
            try:

                if self.use_simple_contrast:
                    contrastive_loss, contrastive_dict = self.contrastive_learner(
                        hyper_out_for_cl,
                        gnn_out_for_cl
                    )
                else:

                    contrastive_loss, contrastive_dict = self.contrastive_learner(
                        hyper_out=hyper_out_for_cl,
                        gnn_out=gnn_out_for_cl,
                        dia_len=dia_len,
                        labels=labels
                    )

                if adho_sparse_loss is not None:
                    if contrastive_loss is None:
                        contrastive_loss = adho_sparse_loss
                    else:
                        contrastive_loss = contrastive_loss + adho_sparse_loss

                if contrastive_loss is not None and torch.rand(1).item() < 0.01:
                    print(f"[CL] Loss: {contrastive_loss.item():.4f}, Components: {contrastive_dict}, adho_sparse: {float(adho_sparse_loss) if adho_sparse_loss is not None else 0.0:.6f}")
            except Exception as e:
                print(f"[对比学习错误] {e}")
                import traceback
                traceback.print_exc()
                contrastive_loss = None
                contrastive_dict = {}

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1, contrastive_loss, contrastive_dict



    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i+3):
                if _ < 3:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            
            Gnodes=[]
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*3

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features