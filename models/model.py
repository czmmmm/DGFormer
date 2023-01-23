import os.path

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, layer, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn_score.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency.to(input_feature.device), support)
        if self.use_bias:
            output += self.bias
        return output


class FeedForwardGCN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.gcn1 = GraphConvolution(dim, mult*dim)
        self.ln1 = nn.LayerNorm(mult*dim)
        self.gcn2 = GraphConvolution(mult*dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dp = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, feature, adj):
        residual = feature
        x = self.gcn1(feature, adj)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.gcn2(x, adj)
        x = self.ln2(x)
        x = self.gelu(x)
        out = residual + x
        return out


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        if len(graph.size()) == 2:
            mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]
        else:
            mul_L = self.batch_cheb_polynomial(L)

        result = torch.matmul(mul_L.to(inputs.device), inputs)  # [K, B, N, C]

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result
    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(-1)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    def batch_cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        b, _, N = laplacian.size()  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, b, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = repeat(torch.eye(N, device=laplacian.device, dtype=torch.float), 'h w -> b h w', b=b)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.matmul(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        size = graph.shape
        if normalize:

            if len(size) > 2:
                D = torch.diag_embed(torch.sum(graph, dim=-1)**(-1/2))
                L = torch.eye(graph.size(-1), device=graph.device, dtype=graph.dtype) - torch.matmul(torch.matmul(D, graph), D)

            else:
                D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
                L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class GraphConv(nn.Module):
    def __init__(self, K, input_dim, output_dim, p_dropout=None):
        super(GraphConv, self).__init__()

        self.gconv = ChebConv(input_dim, output_dim, K)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class ResChebGC(nn.Module):
    def __init__(self, order_m, input_dim, output_dim, hid_dim, p_dropout):
        super(ResChebGC, self).__init__()
        self.gconv1 = GraphConv(order_m, input_dim, hid_dim, p_dropout)
        self.gconv2 = GraphConv(order_m, hid_dim, output_dim, p_dropout)

    def forward(self, x, adj):
        residual = x
        out = self.gconv1(x, adj)
        out = self.gconv2(out, adj)
        return residual + out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, order_m, num_heads, ff_mult, attn_dropout=0., dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.chebgcn1 = ResChebGC(order_m=order_m, input_dim=dim, output_dim=dim, hid_dim=2*dim, p_dropout=dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=ff_mult, dropout=dropout)

        self.chebgcn2 = ResChebGC(order_m=order_m, input_dim=dim, output_dim=dim, hid_dim=2*dim, p_dropout=dropout)

    def sparse_adj(self, x, n, k):

        # GCN for sparse adj
        B, N, _ = x.shape
        dist = torch.sort(torch.cdist(x, x), dim=-1)
        idx = dist.indices[..., 0:k]
        idx_f = torch.flatten(idx, start_dim=1)
        idx_add = torch.arange(0, n * n, n).repeat(k, 1).T.flatten().cuda()
        indx = idx_f + idx_add
        mask = torch.zeros(B, N, N).cuda()
        mask = torch.flatten(mask, start_dim=1)
        new_mask = mask.scatter(1, indx, 1).view(-1, n, n)

        # Normalization
        deg = torch.sum(new_mask, dim=-1)
        deg_sqrt = deg.pow(-0.5)
        deg_diag = torch.diag_embed(deg_sqrt)
        norm = torch.matmul(torch.matmul(deg_diag, new_mask), deg_diag)
        return norm

    def forward(self, layer, x, adj1, dym_k):
        x = x + self.self_attn(layer, self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # Immobile GCN
        x = self.chebgcn1(x, adj1)

        # Dynamic adj
        num_n = adj1.size()[0]
        dym_adj = self.sparse_adj(x, num_n, dym_k)
        x = self.chebgcn2(x, dym_adj)

        return x


class DGFormer(nn.Module):
    def __init__(self,
                 num_joints,
                 order_m,
                 dym_k,
                 adjacency,
                 dim,
                 num_layers,
                 num_heads,
                 ff_mult=4,
                 attn_dropout=0.1,
                 dropout=0.1,
                 drop_path_rate=0.,
                 positional_embedding='learnable'):
        super().__init__()

        self.adj = adjacency
        self.dym_k = dym_k

        positional_embedding = positional_embedding if positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        self.embd_dim = dim

        # Init Embeddings
        self.init_embedding = nn.Linear(2, self.embd_dim)

        # Positional Encoding
        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, num_joints, self.embd_dim), requires_grad=True)
                nn.init.trunc_normal_(self.pos_embedding, std=0.2)
            else:
                self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(num_joints, self.embd_dim), requires_grad=False)
        else:
            self.pos_embedding = None

        self.dropout = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.layers = nn.ModuleList([
            TransformerEncoder(dim=dim, order_m=order_m, num_heads=num_heads, ff_mult=ff_mult,
                               attn_dropout=attn_dropout, dropout=dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3)
        )
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(num_joints, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(num_joints)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')


        # Init Embedding and Positional Encodings
        x = self.init_embedding(x)
        if self.pos_embedding is not None:
            x += self.pos_embedding

        # x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(i, x, self.adj, self.dym_k)

        out = self.head(x)
        out = rearrange(out, '(b f) p c -> b f p c', b=b)

        return out

