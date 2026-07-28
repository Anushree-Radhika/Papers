import torch
import torch.nn as nn
from ScaleDotProductAttention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    #dim_model represents the 512 -> embedding size used to represent a single word
    def __init__(self,dim_model, n_heads):
        
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)
        self.w_concat = nn.Linear(dim_model, dim_model)
        
    
    def forward(self,q,k,v,mask = None):
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        q,k,v = self.split(q),self.split(k),self.split(v)
        out,attention = self.attention(q,k,v, mask = mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out
    
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
        
        
        
        
        
        
        

