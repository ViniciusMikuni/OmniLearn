import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepth(nn.Module):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382).
    
    Reference:
        https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.size(0),) + (1,) * (x.dim() - 1)
            random_tensor = keep_prob + torch.rand(shape, device=x.device)
            random_tensor = torch.floor(random_tensor)
            return x * random_tensor
        return x

class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: int):
        super(RandomDrop, self).__init__()
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.size(0), 1)
            random_tensor = keep_prob + torch.rand(shape, device=x.device)
            random_tensor = torch.floor(random_tensor)
            x[:, :, self.num_skip:] = x[:, :, self.num_skip:] * random_tensor.unsqueeze(-1)
        return x

class SimpleHeadAttention(nn.Module):
    """Simple MHA where masks can be directly added to the inputs."""
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):
        super(SimpleHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, int_matrix=None, mask=None, training=False):
        B, N, C = x.size()
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1))
        
        if int_matrix is not None:
            attn += int_matrix

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn += (1.0 - mask) * -1e9

        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) if training else x
        return x, attn

class TalkingHeadAttention(nn.Module):
    """Talking-head attention as proposed in CaiT: https://arxiv.org/abs/2003.02436."""
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):
        super(TalkingHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, int_matrix=None, mask=None, training=False):
        B, N, C = x.size()
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1))
        if int_matrix is not None:
            attn += int_matrix

        attn = self.proj_l(attn.transpose(1, 2)).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn += (1.0 - mask) * -1e9

        attn = F.softmax(attn, dim=-1)
        
        attn = self.proj_w(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.attn_drop(attn) if training else attn

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) if training else x
        return x, attn

class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(torch.full((projection_dim,), init_values))

    def forward(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * mask
        else:
            return x * self.gamma