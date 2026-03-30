import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

# ── Config ─────────────────────────────────────────────────────────────────────
# vocab_size = 32000

import wandb


vocab_size = tokenizer.vocab_size
d_model    = 256
n_layers   = 6
n_heads    = 8
n_kv_heads = 2
d_ff       = 1024
seq_len    = 256
batch_size = 32
steps      = 5000
lr         = 3e-4


wandb.init(
    project = "gqa-ablation",
    config  = {
        "vocab_size" : vocab_size,
        "d_model"    : d_model,
        "n_layers"   : n_layers,
        "n_heads"    : n_heads,
        "n_kv_heads" : n_kv_heads,
        "d_ff"       : d_ff,
        "seq_len"    : seq_len,
        "batch_size" : batch_size,
        "steps"      : steps,
        "lr"         : lr,
    },
    name = f"gqa_kv{n_kv_heads}"  # each ablation run gets a different name
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"using device: {device}")


# ── RMSNorm ────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.weight * (x / rms)


# ── RoPE ───────────────────────────────────────────────────────────────────────
def build_rope_cache(seq_len, head_dim, device):
    half     = head_dim // 2
    theta    = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(positions, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    half = D // 2
    x1   = x[..., :half]
    x2   = x[..., half:]
    cos  = cos[:T].unsqueeze(0).unsqueeze(0)
    sin  = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ── GQA ────────────────────────────────────────────────────────────────────────
class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = d_model // n_heads
        self.n_rep      = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model,                    bias=False)

        cos, sin = build_rope_cache(seq_len, self.head_dim, device='cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on q and k only
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        # expand k,v so every q head has a matching k,v
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

        # causal mask
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(mask, float('-inf'))
        scores = F.softmax(scores, dim=-1)

        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


# ── SwiGLU FFN ─────────────────────────────────────────────────────────────────
#
# Normal FFN:  x -> Linear -> ReLU -> Linear
# SwiGLU FFN: x -> two Linear projections in parallel
#                   one goes through SiLU (gate)
#                   multiply them together  ← this is the "gating"
#                   then project back down
#
# Why? The gate learns to suppress or amplify parts of the signal
# dynamically per token. Better than a fixed ReLU.

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2   = nn.Linear(d_ff, d_model, bias=False)  # down projection
        self.w3   = nn.Linear(d_model, d_ff, bias=False)  # up projection

    def forward(self, x):
        gate = F.silu(self.w1(x))   # (B, T, d_ff)  ← activated gate
        up   = self.w3(x)           # (B, T, d_ff)  ← raw projection
        return self.w2(gate * up)   # element-wise multiply, then project down


# ── Single Transformer Block ───────────────────────────────────────────────────
#
# Your architecture:
#   RMSNorm -> GQA -> residual add
#   RMSNorm -> SwiGLU -> residual add
#
# Note: norm comes BEFORE the sublayer (pre-norm), not after.
# This is what LLaMA does — it's more stable to train.

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = GQA(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # attention sublayer
        x = x + self.attn(self.norm1(x))   # residual add AFTER attention
        # feedforward sublayer
        x = x + self.ffn(self.norm2(x))    # residual add AFTER ffn
        return x


# ── Full Model ─────────────────────────────────────────────────────────────────
class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, d_ff):
        super().__init__()

        # input: token ids -> float vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff)
            for _ in range(n_layers)
        ])

        # final norm before logits
        self.norm_out = RMSNorm(d_model)

        # project from d_model -> vocab_size to get one score per token
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying: embedding and lm_head share the same weights
        # common in LLMs — saves params, often improves performance
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids, targets=None):
        # token_ids : (B, T)  integer ids
        # targets   : (B, T)  integer ids, shifted by 1 (next token labels)

        x = self.embedding(token_ids)   # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)                # (B, T, d_model)

        x      = self.norm_out(x)       # (B, T, d_model)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        # if targets provided, compute loss right here
        loss = None
        if targets is not None:
            # flatten (B, T, vocab_size) -> (B*T, vocab_size) for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

        return logits, loss


# ── Training Setup ─────────────────────────────────────────────────────────────
model     = LLM(vocab_size, d_model, n_layers, n_heads, n_kv_heads, d_ff).to(device)
nn.init.normal_(model.embedding.weight, mean=0.0, std=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

total_params = sum(p.numel() for p in model.parameters())
print(f"total parameters: {total_params:,}")
x_test = torch.randint(0, vocab_size, (2, 16)).to(device)
with torch.no_grad():
    logits, _ = model(x_test)
    print(f"logits min  : {logits.min().item():.2f}")
    print(f"logits max  : {logits.max().item():.2f}")
    print(f"logits mean : {logits.mean().item():.2f}")


# ── Training Loop ──────────────────────────────────────────────────────────────
# from datasets import load_dataset
# from transformers import AutoTokenizer

# ── load and tokenize once ─────────────────────────────────────────────────────
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

ds     = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text   = "\n".join([x for x in ds["text"] if x.strip()])  # remove blank lines
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
print(f"total tokens: {len(tokens):,}")   # should be ~2M tokens
print(f"tokenizer_number{tokenizer.vocab_size}")
print(f"Tokenizer min size: {(tokens.min().item())}")

# ── get_batch now pulls real slices ───────────────────────────────────────────
def get_batch():
    ix = torch.randint(0, len(tokens) - seq_len, (batch_size,))
    x  = torch.stack([tokens[i   : i+seq_len  ] for i in ix]).to(device)
    y  = torch.stack([tokens[i+1 : i+seq_len+1] for i in ix]).to(device)
    return x, y

model.train()
start = time.time()
for step in range(steps):
    x, y = get_batch()

    logits, loss = model(x, targets=y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops exploding gradients
    optimizer.step()
    scheduler.step()
    if step == 100:
        elapsed = time.time() - start
        per_step = elapsed / 100
        remaining = per_step * (steps - 100)
        print(f"time per step : {per_step:.2f}s")
        print(f"estimated total : {remaining/60:.1f} mins")

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}")


    if step % 10 == 0:
       wandb.log({
        "loss" : loss.item(),
        "lr"   : scheduler.get_last_lr()[0],
        "step" : step,
    })
    print(f"step {step:4d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}", flush=True)

# at the very end of training
wandb.finish()