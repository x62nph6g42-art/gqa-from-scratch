# gqa-from-scratch
README written with assistance from Claude (Anthropic) for clarity and grammar.
___________

Training a LLaMA-style transformer from scratch with Grouped Query Attention (GQA), RoPE, SwiGLU, and RMSNorm — with ablation study on KV heads.
______________
What is this?
____________
This project implements a small LLaMA-style language model from scratch in PyTorch, trained on WikiText-2. The goal is to study how the number of Key-Value heads in Grouped Query Attention (GQA) affects training loss — comparing full Multi-Head Attention (MHA) down to Multi-Query Attention (MQA).

_______

| Stage | Component | Description | Input → Output |
|------|----------|------------|---------------|
| 1 | Input Tokens | Tokenized input sequence | Tokens → IDs |
| 2 | Embedding Layer | nn.Embedding maps token IDs to dense vectors | vocab_size → d_model |
| 3 | Transformer Blocks (× N) | Stack of identical Transformer layers | d_model → d_model |
| 3.1 | RMSNorm | Normalization before attention | d_model → d_model |
| 3.2 | GQA + RoPE | Grouped Query Attention with Rotary Positional Encoding | d_model → d_model |
| 3.3 | Residual Add | Skip connection after attention | d_model + d_model |
| 3.4 | RMSNorm | Normalization before feedforward | d_model → d_model |
| 3.5 | SwiGLU FFN | Feedforward network with SwiGLU activation | d_model → d_ff → d_model |
| 3.6 | Residual Add | Skip connection after FFN | d_model + d_model |
| 4 | Final RMSNorm | Normalization after all Transformer blocks | d_model → d_model |
| 5 | Linear (LM Head) | Projection to vocabulary logits (weight tied with embedding) | d_model → vocab_size |
| 6 | Cross Entropy Loss | Training objective comparing logits with targets | Logits → Loss |

_________

| Category            | Component            | Implementation                     |
|---------------------|----------------------|-------------------------------------|
| Attention           | Self-Attention       | Grouped Query Attention (GQA)       |
| Positional Encoding | Position Encoding    | Rotary Position Embedding (RoPE)    |
| Feedforward         | FFN                  | SwiGLU                              |
| Normalization       | Layer Norm           | RMSNorm (pre-norm)                  |
| Optimization        | Optimizer            | AdamW                               |
| Optimization        | Learning Rate Policy | Cosine Annealing                    |

_______
Config
________
| Parameter   | Value  | Notes                                      |
|------------|--------|--------------------------------------------|
| vocab_size | 32000  | Matched to tokenizer                       |
| d_model    | 256    | Hidden dimension                           |
| n_layers   | 6      | Number of Transformer blocks               |
| n_heads    | 8      | Number of attention heads                  |
| n_kv_heads | 2      | Ablation variable: [8, 4, 2, 1]            |
| d_ff       | 1024   | Feedforward hidden dimension               |
| seq_len    | 256    | Maximum sequence length                    |
| batch_size | 32     | Training batch size                        |
| steps      | 1000   | Training steps                             |
| lr         | 3e-4   | Learning rate                              |
______

Total parameters: ~13.9M
_______

________

| Property     | Value                                      |
|--------------|--------------------------------------------|
| Dataset      | WikiText-2                                 |
| Source       | wikitext-2-raw-v1 (HuggingFace)            |
| Tokenizer    | huggyllama/llama-7b (BPE, vocab=32000)     |
| Total Tokens | ~2.8M                                      |
| Split Used   | Train                                      |

_____
### KV Head Ablation Study

| Variant        | n_kv_heads | KV Cache Savings | Loss @ 100 | Loss @ 1000 |
|----------------|------------|------------------|------------|-------------|
| MHA (baseline) | 8          | —                | 6.87       | 5.07        |
| GQA            | 2          | 75%              | 6.52       | 5.03        |
| MQA            | 1          | 87.5%            | 6.66       | 5.11        |

____

### Key Findings

- GQA (n_kv_heads = 2) achieves lower final loss than MHA (5.03 vs 5.07) while reducing KV cache memory by 75%, indicating a strictly better efficiency–performance trade-off.

- MQA (n_kv_heads = 1) incurs only a minor degradation (+0.04 loss vs MHA) despite an 87.5% reduction in KV cache size.

- All configurations exhibit comparable convergence, suggesting KV head count has limited influence on optimization dynamics at this scale.

- These results are consistent with prior empirical findings on GQA.

__________

### KV Cache Memory Reduction

The relative KV cache memory savings when using GQA/MQA is:

savings = 1 - (n_kv_heads / n_heads)

#### Examples (n_heads = 8)

| n_kv_heads | Calculation                | Savings |
|------------|---------------------------|---------|
| 2          | 1 - (2 / 8) = 1 - 0.25    | 75%     |
| 1          | 1 - (1 / 8) = 1 - 0.125   | 87.5%   |

______

### Key Findings

- **GQA (n_kv_heads = 2)** outperforms MHA (n_kv_heads = 8):  
  final loss **5.03 vs 5.07**, with **75% KV cache memory reduction**.

- **MQA (n_kv_heads = 1)** remains competitive:  
  only **+0.04 loss vs MHA**, while achieving **87.5% memory savings**.

- **Convergence is similar across all variants**:  
  at this scale, varying KV head count has **minimal impact on final loss**.

  ______
### Setup

```bash
git clone https://github.com/x62nph6g42-art/gqa-from-scratch
cd gqa-from-scratch

# install dependencies
pip install torch datasets transformers wandb
```
  _____

  ### References

- *Attention Is All You Need* — Vaswani et al., 2017  
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., 2023


---

### Learning Resources

- Claude — used for understanding GQA, RoPE, and SwiGLU concepts. 
- Gemini — supplementary reference for GQA concepts  





