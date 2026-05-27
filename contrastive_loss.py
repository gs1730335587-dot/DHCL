import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphContrastiveLearning(nn.Module):

    def __init__(self, hidden_dim, temperature=0.07):
        super(DualGraphContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        
        # 投影头 (projection heads) - 关键组件
        self.hyper_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        self.gnn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
    def forward(self, hyper_out, gnn_out, dia_len, labels):

        inter_loss = self.inter_graph_contrast(hyper_out, gnn_out)
        intra_hyper_loss = self.intra_graph_contrast(hyper_out, labels, dia_len, use_gnn_proj=False)
        intra_gnn_loss = self.intra_graph_contrast(gnn_out, labels, dia_len, use_gnn_proj=True)
        cross_loss = self.cross_dialog_contrast(hyper_out, gnn_out, dia_len, labels)
        
        # 加权求和
        total_loss = inter_loss + 0.5 * (intra_hyper_loss + intra_gnn_loss) + 0.3 * cross_loss
        
        loss_dict = {
            'inter': inter_loss.item(),
            'intra_hyper': intra_hyper_loss.item(),
            'intra_gnn': intra_gnn_loss.item(),
            'cross': cross_loss.item()
        }
        
        return total_loss, loss_dict
    
    def inter_graph_contrast(self, hyper_out, gnn_out):

        z_hyper = F.normalize(self.hyper_proj(hyper_out), dim=1)  # [N, 128]
        z_gnn = F.normalize(self.gnn_proj(gnn_out), dim=1)        # [N, 128]
        
        N = z_hyper.size(0)
        

        logits = torch.mm(z_hyper, z_gnn.T) / self.temperature  # [N, N]

        labels = torch.arange(N).cuda()
        

        loss_h2g = F.cross_entropy(logits, labels)
        loss_g2h = F.cross_entropy(logits.T, labels)
        
        return (loss_h2g + loss_g2h) / 2
    
    def intra_graph_contrast(self, out, emotion_labels, dia_len, use_gnn_proj=False):

        proj = self.gnn_proj if use_gnn_proj else self.hyper_proj
        z = F.normalize(proj(out), dim=1)  # [N, 128]
        

        mask = self.build_intra_mask(emotion_labels, dia_len)  # [N, N]
        

        sim_matrix = torch.mm(z, z.T) / self.temperature  # [N, N]
        

        sim_matrix_exp = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
        

        diag_mask = torch.eye(mask.size(0), dtype=torch.bool, device=mask.device)
        mask = mask * (~diag_mask).float()
        

        pos_sum = (sim_matrix_exp * mask).sum(1)
        all_sum = sim_matrix_exp.sum(1)
        

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        

        has_pos = mask.sum(1) > 0
        if has_pos.sum() > 0:
            loss = loss[has_pos].mean()
        else:
            loss = torch.tensor(0.0).cuda()
        
        return loss
    
    def cross_dialog_contrast(self, hyper_out, gnn_out, dia_len, labels):

        fused = (hyper_out + gnn_out) / 2
        z = F.normalize(self.hyper_proj(fused), dim=1)  # [N, 128]
        

        mask = self.build_cross_dialog_mask(labels, dia_len)  # [N, N]
        
        if mask.sum() == 0:

            return torch.tensor(0.0).cuda()
        

        sim_matrix = torch.mm(z, z.T) / self.temperature
        sim_matrix_exp = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
        

        pos_sum = (sim_matrix_exp * mask).sum(1)
        all_sum = sim_matrix_exp.sum(1)
        
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        

        has_pos = mask.sum(1) > 0
        if has_pos.sum() > 0:
            loss = loss[has_pos].mean()
        else:
            loss = torch.tensor(0.0).cuda()
        
        return loss
    
    def build_intra_mask(self, emotion_labels, dia_len):

        N = emotion_labels.size(0)
        mask = torch.zeros(N, N).cuda()
        
        start = 0
        for length in dia_len:
            end = start + length

            dialog_labels = emotion_labels[start:end]
            

            for i in range(length):
                for j in range(length):
                    if i != j and dialog_labels[i] == dialog_labels[j]:
                        mask[start+i, start+j] = 1
            
            start = end
        
        return mask
    
    def build_cross_dialog_mask(self, emotion_labels, dia_len):

        N = emotion_labels.size(0)
        mask = torch.zeros(N, N).cuda()
        

        dialog_ids = torch.zeros(N, dtype=torch.long).cuda()
        start = 0
        for dialog_id, length in enumerate(dia_len):
            end = start + length
            dialog_ids[start:end] = dialog_id
            start = end
        

        for i in range(N):
            for j in range(N):
                if dialog_ids[i] != dialog_ids[j] and emotion_labels[i] == emotion_labels[j]:
                    mask[i, j] = 1
        
        return mask

