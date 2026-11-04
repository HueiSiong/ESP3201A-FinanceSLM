"""Original Transformer family with numerically safe mixed-precision operations."""
import torch
from model import FinanceSLM as OriginalFinanceSLM, RMSNorm, RotaryEmbedding

class StableRMSNorm(RMSNorm):
    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight

class StableRotaryEmbedding(RotaryEmbedding):
    def forward(self, x):
        # Compute angles/rotation in float32, then match V's autocast dtype for SDPA.
        return super().forward(x.float()).to(x.dtype)

class FinanceSLM(OriginalFinanceSLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dim = self.token_embedding.embedding_dim
        self.norm = StableRMSNorm(dim)
        for block in self.blocks:
            block.attn_norm = StableRMSNorm(dim)
            block.ffn_norm = StableRMSNorm(dim)
            block.attn.rope = StableRotaryEmbedding(block.attn.head_dim)
