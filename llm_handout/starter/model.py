# """A small GPT in plain PyTorch. Yours to modify or replace entirely —
# attention, SSM, whatever — as long as evaluate.py still works and the
# parameter cap holds.
# """
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Config:
#     vocab_size = 2048      # custom BPE tokenizer
#     block_size = 128
#     n_layer = 4
#     n_head = 4
#     n_embd = 160
#     dropout = 0.0
#     tie_weights = False   # <- one of many things worth questioning


# class SelfAttention(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.n_head = cfg.n_head
#         self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
#         self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
#         self.drop = nn.Dropout(cfg.dropout)

#     def forward(self, x):
#         B, T, C = x.shape
#         q, k, v = self.qkv(x).split(C, dim=2)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.drop(self.proj(y))


# class Block(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(cfg.n_embd)
#         self.attn = SelfAttention(cfg)
#         self.ln2 = nn.LayerNorm(cfg.n_embd)
#         self.mlp = nn.Sequential(
#             nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
#             nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x


# class GPT(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
#         self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
#         self.drop = nn.Dropout(cfg.dropout)
#         self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
#         self.ln_f = nn.LayerNorm(cfg.n_embd)
#         self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
#         if cfg.tie_weights:
#             self.head.weight = self.tok_emb.weight
#         self.apply(self._init)

#     def _init(self, m):
#         # baseline init: plain normal, one std for everything
#         if isinstance(m, (nn.Linear, nn.Embedding)):
#             nn.init.normal_(m.weight, mean=0.0, std=0.05)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
#         pos = torch.arange(T, device=idx.device)
#         x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
#         for blk in self.blocks:
#             x = blk(x)
#         logits = self.head(self.ln_f(x))
#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
#                                    targets.reshape(-1))
#         return logits, loss

#     def n_params(self):
#         return sum(p.numel() for p in self.parameters())
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 2048      # MUST match your BPE tokenizer vocab size
    block_size = 256       # 256 context window
    n_layer = 4            
    n_head = 4             # 4 Query heads
    n_embd = 216           # Bumped up to 216! (Because MQA saves params)
    dropout = 0.0          # Keep at 0!
    tie_weights = True     # Share embedding and output head

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def apply_rope(x):
    B, T, H, D = x.shape
    pos = torch.arange(T, device=x.device).float()
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, D, 2, device=x.device).float() / D)
    theta = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(theta).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(2)
    
    x1, x2 = x.chunk(2, dim=-1)
    rx = torch.cat([-x2, x1], dim=-1)
    return x * cos.repeat(1, 1, 1, 2) + rx * sin.repeat(1, 1, 1, 2)

class MultiQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        
        # MQA: 1 full matrix for Queries, but only 1 tiny head for Keys and Values
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.out_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, 1, self.head_dim) # 1 shared head
        v = self.v_proj(x).view(B, T, 1, self.head_dim) # 1 shared head

        q = apply_rope(q).transpose(1, 2)
        k = apply_rope(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch SDPA handles the broadcasting of the 1 K/V head to the 4 Q heads
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class SwiGLUBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = MultiQueryAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        
        # SwiGLU: Parameter-matched to standard 4x MLP
        hidden_dim = int((8 / 3) * cfg.n_embd) 
        self.w1 = nn.Linear(cfg.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, cfg.n_embd, bias=False)
        self.w3.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        # SwiGLU activation
        h_swiglu = F.silu(self.w1(h)) * self.w2(h)
        x = x + self.w3(h_swiglu)
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # No pos_emb matrix needed (RoPE handles it)
        self.blocks = nn.ModuleList(SwiGLUBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - (self.tok_emb.weight.numel() if self.cfg.tie_weights else 0)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            std = 0.02
            if hasattr(m, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        # Calculate logits for ALL tokens in the sequence
        logits = self.head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        
        return logits, None