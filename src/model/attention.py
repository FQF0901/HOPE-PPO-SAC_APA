

import torch
from torch import nn
from einops import rearrange
from tqdm import trange

'''
AttentionNetwork
├── encoder (Transformer)
│   ├── layers (nn.ModuleList) [depth 层]
│   │   └── [i-th Layer]
│   │       ├── PreNorm
│   │       │   ├── LayerNorm (nn.LayerNorm) [dim]
│   │       │   └── res
│   │       │       └── attn (Attention) + x
│   │       │           ├── to_qkv (nn.Linear) [dim -> inner_dim * 3]
│   │       │           ├── q, k, v (rearrange) [b, n, (h d) -> b, h, n, d]
│   │       │           ├── dots (torch.matmul) [b, h, n, n]
│   │       │           ├── attn (nn.Softmax) [b, h, n, n]
│   │       │           ├── out (torch.matmul) [b, h, n, d]
│   │       │           └── to_out (nn.Sequential) [inner_dim -> dim]
│   │       │               ├── nn.Linear [inner_dim -> dim]
│   │       │               └── nn.Dropout [dropout]
│   │       └── PreNorm
│   │           ├── LayerNorm (nn.LayerNorm) [dim]
│   │           └── res
│   │               └── ff (FeedForward) + x
│   │                   ├── nn.Linear [dim -> hidden_dim]
│   │                   ├── nn.Tanh
│   │                   ├── nn.Dropout [dropout]
│   │                   ├── nn.Linear [hidden_dim -> dim]
│   │                   └── nn.Dropout [dropout]
│   └── ... (重复 depth 层)
├── rearrange [b, n, d -> b, (n d)]
└── output (nn.Sequential)
    ├── nn.Linear [(n_features * dim) -> hidden_dim]
    ├── nn.Tanh
    └── nn.Linear [hidden_dim -> output_dim]
'''

class PreNorm(nn.Module):
    """
    在应用特定函数之前执行层归一化的模块。
    
    该类结合了层归一化和任意给定的函数，以便在神经网络的前向传播过程中，
    先对输入数据进行层归一化，然后再应用给定的函数。
    
    属性:
    - dim: 输入数据的维度，用于层归一化。
    - fn: 一个函数或模块，用于在归一化后应用于输入数据。
    """
    
    def __init__(self, dim, fn):
        """
        初始化PreNorm类。
        
        参数:
        - dim: 输入数据的维度，用于层归一化。
        - fn: 一个函数或模块，用于在归一化后应用于输入数据。
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化，对输入数据的每个样本的最后一个维度进行归一化
        self.fn = fn  # 保存传入的函数或模块，以便在前向传播时使用

    def forward(self, x, **kwargs):
        """
        前向传播方法。
        
        先对输入数据进行层归一化，然后应用传入的函数或模块。
        这种结构有助于在深度学习模型中集成归一化操作，从而可能提高模型的性能。
        
        参数:
        - x: 输入数据，将对其进行归一化和函数应用。
        - **kwargs: 允许额外的关键字参数传递给传入的函数或模块。
        
        返回:
        - 经过归一化和函数应用后的数据。
        """
        # 先进行层归一化
        normalized_x = self.norm(x)
        # 然后应用传入的函数或模块，并传递额外的关键字参数
        return self.fn(normalized_x, **kwargs)

class Attention(nn.Module):
    """
    实现一个注意力机制模块。
    
    参数:
        dim (int): 输入和输出的维度。
        heads (int, optional): 注意力头的数量，默认为8。
        dim_head (int, optional): 每个注意力头的维度，默认为64。
        dropout (float, optional): Dropout的概率，默认为0。
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # 计算内部维度，即所有注意力头维度的总和
        inner_dim = dim_head *  heads
        # 确定是否需要进行输出投影，如果头数为1且头维度与输入维度相同，则不需要
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # 计算缩放因子，用于注意力得分的缩放
        self.scale = dim_head ** -0.5

        # 使用softmax函数沿最后一个维度计算注意力权重
        self.attend = nn.Softmax(dim = -1)
        # dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 线性层，用于生成查询、键和值
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 输出层，如果不需要进行输出投影，则使用Identity
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为[b, n, f]，其中b是批次大小，n是序列长度，f是特征维度。
        
        返回:
            Tensor: 输出张量，形状与输入张量相同。
        """
        # x: [b, n, f]
        # 生成查询、键和值，并将它们拆分成三个独立的张量
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 对查询、键和值进行重新排列，以便进行注意力计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 计算注意力得分，并进行缩放
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b, h, n, n]

        # 使用softmax函数计算注意力权重，并应用dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # 使用注意力权重对值进行加权求和
        out = torch.matmul(attn, v) # v: [b, h, n, d]   attn: [b, h, n, n]
        # 对输出进行重新排列，以便进行输出投影
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 应用输出投影（如果有）
        return self.to_out(out) # [b, n, h*d] -> [b, n, f]
    
class FeedForward(nn.Module):
    """
    FeedForward神经网络类，包含两个线性变换层，用于前馈神经网络的实现。
    
    该类初始化时构建了一个简单的前馈神经网络，包括一个输入层到隐藏层的变换，
    后跟一个非线性激活函数Tanh，一个Dropout层用于防止过拟合，然后是隐藏层到输出层的变换，
    最后跟一个Dropout层。
    
    参数:
    dim (int): 输入和输出层的维度。
    hidden_dim (int): 隐藏层的维度。
    dropout (float): Dropout的概率，默认为0.。
    
    属性:
    net (nn.Sequential): 一个序列容器，定义了前馈神经网络的结构。
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # 使用Tanh作为非线性激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        前馈神经网络的向前传播函数。
        
        参数:
        x (Tensor): 输入张量。
        
        返回:
        Tensor: 经过神经网络变换后的输出张量。
        """
        return self.net(x)

class Transformer(nn.Module):
    """
    Transformer模型，由多层Attention和FeedForward组成。
    
    参数:
    - dim: 输入和输出的维度。
    - depth: Transformer块的层数。
    - heads: 多头注意力机制的头数。
    - dim_head: 每个头的维度。
    - mlp_dim: FeedForward网络中的中间层维度。
    - dropout: Dropout的概率，默认为0.。
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # 初始化Transformer层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 每层包含一个Attention模块和一个FeedForward模块，都使用PreNorm进行预归一化
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        """
        Transformer的前向传播。
        
        参数:
        - x: 输入序列。
        
        返回:
        - x: 经过Transformer编码后的序列。
        """
        # 逐层进行前向传播
        for attn, ff in self.layers:
            # 使用Attention机制并进行残差连接
            x = attn(x) + x
            # 使用FeedForward网络并进行残差连接
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


    