

import torch
from torch import nn
from einops import rearrange
from tqdm import trange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x: [b, n, f]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b, h, n, n]

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # v: [b, h, n, d]   attn: [b, h, n, n]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # [b, n, h*d] -> [b, n, f]
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class AttentionNetwork(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, n_features, hidden_dim, output_dim):
        '''
        dim: 模型中特征向量的维度
        depth: Transformer模型的深度（层数）
        heads: Transformer模型中自注意力机制头的数量
        dim_head: 每个注意力头的维度
        mlp_dim: Transformer中全连接层的隐藏单元数

        n_features: 输入特征的数量
        hidden_dim: 输出层中隐藏层的维度
        output_dim: 输出的维度
        '''
        super().__init__()
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim,)
        self.output = nn.Sequential(
            nn.Linear(n_features*dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),
        )
        # 定义了一个可学习的参数 view_embed，它是一个三维的张量，具体形状是 (1, n_features, dim)
        # nn.Parameter 将张量转换为模型的可训练参数，意味着在训练过程中，PyTorch会自动更新和优化 view_embed 的值，使其能够最大化地帮助模型达到任务优化的目标
        # 总之，view_embed 参数的作用是通过在模型中引入可学习的先验信息或者额外的视角，帮助模型更有效地处理输入数据，提高模型的泛化能力和性能
        self.view_embed = nn.Parameter(torch.zeros(1, n_features, dim))

    def forward(self, x):
        # x = x + self.view_embed
        x = self.encoder(x) # 首先，输入x通过self.encoder进行编码处理，得到的结果保存在x中
        x = rearrange(x, 'b n d -> b (n d)')    # 然后，通过rearrange函数将编码后的张量x从形状 'b n d' 重新排列为 'b (n d)'。这一步通常是为了将多维的特征张量展平成一维，以便送入全连接层进行处理
        return self.output(x)


    