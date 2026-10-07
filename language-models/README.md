# Language Models

[Back to portfolio](../README.md)

## [Masked Language Modeling](01_masked_language_modeling.ipynb)

Tokenization, masking, self-attention and a compact RoBERTa-style model.

## [RoBERTa Pretraining](02_roberta_pretraining.ipynb)

Data preparation, modular encoder components, masked language modeling and Accelerate training code.

## Environment and status

Requires PyTorch, transformers, datasets, accelerate, safetensors, torchmetrics, NumPy, matplotlib, and tqdm. The pretraining notebook contains sections intended for separate Python modules and a shell script. Extract the modules and configure dataset/cache paths before attempting a training run.

Open the notebook in JupyterLab or a suitable GPU notebook environment and inspect its configuration before execution. These are learning implementations; some cells are incomplete or require corrections. See [validation notes](../VALIDATION.md).

Original upload titles are recorded in [SOURCES.md](../SOURCES.md). Exact tutorial/repository attribution and the distinction between reproduced code and personal extensions remain to be documented by the author.

## [Llama 2 from Scratch](03_llama2_from_scratch.ipynb)

Decoder architecture, RMSNorm, rotary embeddings, grouped-query attention, KV cache and text generation.

Requires PyTorch, SentencePiece and tqdm. Assemble the referenced model module and configure checkpoint/tokenizer paths. The default inference example expects Llama 2 weights; no weights are included.
