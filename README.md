# MicroGPT

## Model architecture

With the current 46-token vocabulary, the default MicroGPT model has:

- 4,800 trainable parameters
- 1 Transformer layer
- 4 attention heads
- 16-dimensional token embeddings
- 4 dimensions per attention head (`16 / 4`)

These defaults are defined by `ModelConfig` in `model.py`. The exact parameter
count changes if the vocabulary size or model configuration changes.
