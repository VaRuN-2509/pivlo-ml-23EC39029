# Run 1: Baseline
- **Hypothesis:** Run starter code as provided to establish a baseline.
- **Changes:** None.
- **train_loss:** 1.7315
- **BPB:** 2.3718

# Run 2: Tokenizer Bottleneck & Custom BPE
- **Hypothesis:** The baseline byte-level tokenizer forces the model to waste its strict 2,000-step budget predicting deterministic UTF-8 continuation bytes for Hindi text (which takes 3 bytes per character). Furthermore, it artificially shrinks the semantic context window. Training a custom Byte-Pair Encoding (BPE) tokenizer will compress these sequences, allowing the model to see much more actual text within its 128-token context window and allocate its parameter budget toward learning real language structure rather than raw byte encoding.
- **Changes:** 1. Replaced `ByteTokenizer` with a custom, highly optimized BPE Tokenizer.
  2. Trained the tokenizer purely on `train_corpus.txt` with a target vocabulary size of 2048.
  3. Ensured lossless byte-level fallback for out-of-vocab characters to pass the round-trip check.
  4. Updated `vocab_size = 2048` in `model.py` `Config` to match the generated BPE vocabulary. Model architecture otherwise unchanged.
- **BPB Before:** 2.3718
- **BPB After:** 2.1199
- **Conclusion:** [e.g., The custom BPE tokenizer drastically reduced the sequence length for the Hindi portion of the corpus. Without changing the neural network's depth or attention mechanisms, simply compressing the input representations resulted in a massive drop in Bits Per Byte.]


# Run 3: Optimizer and Learning Rate Optimization
- **Hypothesis:** The baseline uses standard Adam with a constant learning rate, which causes instability early in training and fails to settle into a low-loss basin at the end. By implementing AdamW (for weight decay), a cosine decay schedule with linear warmup, and gradient clipping, the model will train much more stably and efficiently within the 2,000-step limit. Increasing the batch size to 32 will also maximize the data seen per step.
- **Changes:** 1. Replaced `torch.optim.Adam` with `torch.optim.AdamW` (weight decay 0.1).
  2. Implemented a linear warmup (200 steps) to a peak LR of 1e-3, followed by a cosine decay to 1e-4.
  3. Added `torch.nn.utils.clip_grad_norm_` (max_norm=1.0) before the optimizer step.
  4. Passed `--batch 16` via command line to increase tokens per step.
- **BPB Before:** 2.1199
- **BPB After:** 1.811



# Run 4: Optimizer and Learning Rate Optimization

  4. Passed `--batch 32` via command line to increase tokens per step.
- **BPB Before:** 1.811
- **BPB After:** 1.7877



# Run 5: Architecture Modernization (Parameter Tetris)
- **Hypothesis:** The baseline GPT architecture wastes precious parameter budget on absolute positional embeddings and standard multi-head attention. By adopting modern LLM standards (MQA, RoPE, SwiGLU, RMSNorm), we can delete hundreds of thousands of "wasted" parameters and reinvest them into a wider embedding dimension, maximizing representational power within the strict 2,000,000 parameter limit without increasing CPU compute time.
- **Changes:** 1. Implemented Multi-Query Attention (MQA) to drastically reduce the parameter overhead of the attention `qkv` projections.
  2. Replaced standard GeLU MLPs with SwiGLU blocks, matching the parameter count by using an `8/3` hidden dimension ratio.
  3. Replaced `LayerNorm` with `RMSNorm` (faster CPU compute, removes bias parameters).
  4. Replaced learned absolute positional embeddings with Rotary Positional Embeddings (RoPE), adding a `RoPECache` to prevent trigonometric math from bottlenecking the laptop CPU.
  5. Reinvested the saved parameters by increasing `n_embd` from 160 to 216, pushing the total model size to ~1.96M (safely under the 2M cap).
- **BPB Before:** 1.7877
- **BPB After:** 1.6721


# Run 6: Gradient Accumulation (The Batch Size Loophole)
- **Hypothesis:** The assignment strictly caps training at 2,000 *optimizer steps*, but does not limit the number of tokens processed per step. However, increasing the physical batch size causes the laptop CPU to run out of RAM. By implementing Gradient Accumulation, we can run multiple small micro-batches, sum their gradients, and take one massive optimizer step. This simulates a huge batch size (improving gradient stability and convergence) without blowing up memory requirements or violating the 2,000-step rule.
- **Changes:** 1. Wrapped the forward and backward passes in a micro-batch `for` loop (`grad_accum_steps = 4`).
  2. Divided the loss by `grad_accum_steps` before calling `loss.backward()` to mathematically match the scale of a true large batch.
  3. Moved `opt.step()`, `opt.zero_grad()`, and `torch.nn.utils.clip_grad_norm_` outside the loop so they only trigger once per full accumulation cycle.
  4. Used a physical batch size of 16 to keep CPU memory safe, resulting in an effective batch size of 64 (16 × 4) per step.
- **BPB Before:** 1.6721
- **BPB After:** 1.6425
- **Conclusion:** [e.g., Gradient accumulation successfully decoupled our effective batch size from our hardware memory limits. By feeding the model 4x more data before every weight update, the gradients were much less noisy. The model squeezed significantly more learning out of its strict 2,000-step budget, resulting in our lowest BPB yet.]