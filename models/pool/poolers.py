import torch
from torch import optim, nn, utils, Tensor
from timm.layers import Mlp
from einops import rearrange

# LUNA Implementation
class AttentivePoolingWithLearnedQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim) 
        self.reconstruction_shape = self.input_dim
        self.decoder_attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=0.15)
        self.decoder_ffn = Mlp(in_features=self.embed_dim, hidden_features=int(self.embed_dim*4), out_features=embed_dim, act_layer=nn.GELU, drop=0.15)
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)
    
    def forward(self, x):
        """
        Output shape:
            [B, num_tokens, in_chans, input_dim]
        Args:
            x: [B, num_tokens+1, embed_dim]
            channel_embeddings: [B, in_chans, embed_dim]
        """
        # B, num_patches, embed_dim = x.shape
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)

        x = self.decoder_attn(query=decoder_queries, key=x, value=x)[0]
        x = x[:,0,:]
        x = self.decoder_ffn(x)
        
        return x

class Pooler(nn.Module):
    def __init__(self, pool_method="mean"):
        super().__init__()
        
        methods = {
            "max": self.max_pool,
            "mean": self.mean_pool,
            "concat": self.concat
        }

        if pool_method not in methods:
            raise ValueError(f"Method {pool_method} not supported. Choose from {list(methods.keys())}")

        self.pool_fn = methods[pool_method]

    @staticmethod
    def max_pool(x):
        return x.max(dim=1)[0]

    @staticmethod
    def mean_pool(x):
        return x.mean(dim=1)
    
    @staticmethod
    def concat(x):
        return rearrange(x, 'b n e -> b (n e)')
    
    def forward(self, x):
        return self.pool_fn(x)

class AttentiveDelta(nn.Module):
    # Input : Model Embeddings for two datapoints
    # Output : Pooled Embedding

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.Pooler = AttentivePoolingWithLearnedQueries(input_dim, embed_dim, num_heads)

    def forward(self, x_0, x_1):
        delta = self.Pooler(x_1) - self.Pooler(x_0)
        return delta                                                                       

class NNDelta(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (0, 256, 256),
        embed_dim: int = 256,
        pool_method: str = "mean",
        drop=0.15
    ):
        super().__init__()

        self.Pooler = Pooler(pool_method=pool_method)

        if pool_method == "concat":
            self.input_dim = input_shape[2] * input_shape[1]
        else:
            self.input_dim = input_shape[2]

        self.mlp = Mlp(in_features=self.input_dim, hidden_features=self.input_dim*4, out_features=embed_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x_0, x_1):
        delta = self.Pooler(x_1) - self.Pooler(x_0)
        return delta

class Delta(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_0, x_1):
        return x_1 - x_0