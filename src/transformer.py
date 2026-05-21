import torch as t
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Float, Int
from dataclasses import dataclass

@dataclass
class Config:
    d_model:int = 128
    d_vocab:int = 50257
    n_ctx: int = 128
    d_head: int = 32
    d_mlp: int = 512
    n_heads: int = 4
    n_layers: int = 4
    range_init: float=0.02
    use_rope:bool = True
    rope_base:int = 10_000

class LayerNorm(nn.Module):
    def __init__(self, cfg:Config, eps=1e-5):
        super().__init__()
        self.cfg = cfg
        self.eps = eps
        self.gamma = nn.Parameter(t.ones(cfg.d_model))
        self.beta = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(self, x):
        # x.shape = batch, seq, d_model
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        normalized = (x - mean)/t.sqrt(var + self.eps) # batch, seq, d_model
        
        out = normalized * self.gamma + self.beta
        return out

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.range_init)

    def forward(self, tokens: Int[Tensor, "batch position"]):
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_P = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_P, std=cfg.range_init)

    def forward(self, tokens: Int[Tensor, "batch position"]):
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_P[:seq_len], "seq d_model -> batch seq d_model", batch=batch)
    
class RoPE(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_ctx = cfg.n_ctx
        self.inv_freq = cfg.rope_base**(-2*t.arange(0, cfg.d_model, 2)/self.d_model)
        self._build_cache()

    def _build_cache(self):
        seq = t.arange(self.n_ctx).float()
        freqs = einops.einsum(seq, self.inv_freq, 'i, j -> i j')

        emb = t.cat((freqs, freqs), dim=-1) # duplication

        self.cos_cache = emb.cos()[None, None, :, :]
        self.sin_cache = emb.sin()[None, None, :, :]

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return t.stack((-x2, x1), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.range_init)
        nn.init.normal_(self.W_K, std=self.cfg.range_init)
        nn.init.normal_(self.W_V, std=self.cfg.range_init)
        nn.init.normal_(self.W_O, std=self.cfg.range_init)
        if self.cfg.use_rope:
            self.rope = RoPE(cfg)

    def _apply_rope(self, q, k):
        b, seq_len, n_heads, d_head = q.shape
        device = q.device
        cos = self.rope.cos_cache[:, :, :seq_len, :].to(device)
        sin = self.rope.sin_cache[:, :, :seq_len, :].to(device)

        q_flat = q.view(b, seq_len, n_heads*d_head)
        k_flat = k.view(b, seq_len, n_heads*d_head)

        q_rot = (q_flat*cos) + (self.rope.rotate_half(q_flat)*sin)
        k_rot = (k_flat*cos) + (self.rope.rotate_half(k_flat)*sin)
        return q_rot.view(b, seq_len, n_heads, d_head), k_rot.view(b, seq_len, n_heads, d_head)

    def _apply_causal_masking(self, attn_weights:Float[Tensor, "batch n_heads seqQ seqK"]):
        seq = attn_weights.shape[-1]
        mask = t.tril(t.ones(seq, seq, device=attn_weights.device), diagonal=0)
        return attn_weights.where(mask==1, -t.inf)

    def forward(self, x):
        # x.shape = (batch, seq, d_model)
        batch, seq, _ = x.shape
        Q = einops.einsum(x, self.W_Q, "b s d, n d h -> b s n h") + self.b_Q

        K = einops.einsum(x, self.W_K, "b s d, n d h -> b s n h") + self.b_K
        
        if self.cfg.use_rope:
            Q, K = self._apply_rope(Q, K)

        attn_weights = einops.einsum(Q, K, "b sQ n d, b sK n d -> b n sQ sK") /(self.cfg.d_head**0.5) # (batch, n_heads, seq_Q, seq_K)
        
        # Causal masking of the weights
        masked_weights = self._apply_causal_masking(attn_weights)
        attn_scores = masked_weights.softmax(dim=-1)

        V = einops.einsum(x, self.W_V, "b s d, n d h -> b s n h") + self.b_V

        z = einops.einsum(attn_scores, V, "b n sQ s, b s n h -> b n sQ h")

        result = einops.einsum(z, self.W_O, "b n s h, n h d -> b s n d")

        return result.sum(dim=-2)
    
class MLP(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Linear(self.cfg.d_model, self.cfg.d_mlp)
        self.W_out = nn.Linear(self.cfg.d_mlp, self.cfg.d_model)
        self.gelu = nn.GELU()

    def forward(self, resid):
        x = self.W_in(resid)
        x = self.gelu(x)
        x = self.W_out(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid):
        resid = resid + self.attn(self.ln1(resid))
        resid = resid + self.mlp(self.ln2(resid))
        return resid
        

class Unembed(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(cfg.d_model, cfg.d_vocab))
        nn.init.normal_(self.W_U, std=self.cfg.range_init)
    
    def forward(self, x):
        return x @ self.W_U
    

class Transformer(nn.Module):
    def __init__(self, tokenizer, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        residual = self.embed(tokens)
        if not self.cfg.use_rope:
            residual += self.pos_embed(tokens)
        residual = self.blocks(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits
