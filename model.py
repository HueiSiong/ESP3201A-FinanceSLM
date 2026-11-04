
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even.")

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float()
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )

    def forward(self, x):
        seq_len = x.size(-2)

        positions = torch.arange(
            seq_len, device=x.device
        ).float()

        angles = torch.outer(
            positions, self.inv_freq
        )

        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(
                batch, seq_len,
                self.n_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q = self.rope(q)
        k = self.rope(k)

        attention = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )

        attention = attention.transpose(
            1, 2
        ).contiguous().view(batch, seq_len, dim)

        return self.out_proj(attention)


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = int(8 * dim / 3)
        hidden = ((hidden + 31) // 32) * 32

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(
            F.silu(self.gate(x)) * self.up(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FinanceSLM(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        dim=192,
        n_layers=4,
        n_heads=4,
        max_seq_len=128
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)

        self.lm_head = nn.Linear(
            dim, vocab_size, bias=False
        )

        # Share input/output embedding weights
        self.lm_head.weight = self.token_embedding.weight
        # Initialize model weights with smaller values
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids, targets=None):
        batch, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence exceeds context length."
            )

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss


if __name__ == "__main__":
    model = FinanceSLM()

    params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {params:,}")
    print(f"Model size: {params / 1e6:.2f}M")

    # Test a forward pass
    x = torch.randint(0, 8000, (2, 128))

    logits, loss = model(x, x)

    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Forward pass successful!")
