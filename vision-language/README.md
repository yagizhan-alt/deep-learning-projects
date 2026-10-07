# Vision-Language Models and Mixture of Experts

[Back to portfolio](../README.md)

## [Llama 4-style VLM and MoE](llama4_vlm_moe.ipynb)

Text and vision components, rotary embeddings, KV cache, expert routing and multimodal projection.

## Environment and status

Requires PyTorch. This is an incomplete architecture study. The final conditional-generation class references undefined class names and lacks a forward method. Large default configuration values require substantial memory; use small dimensions for future component tests.

Open the notebook in JupyterLab or a suitable GPU notebook environment and inspect its configuration before execution. These are learning implementations; some cells are incomplete or require corrections. See [validation notes](../VALIDATION.md).

Original upload titles are recorded in [SOURCES.md](../SOURCES.md). Exact tutorial/repository attribution and the distinction between reproduced code and personal extensions remain to be documented by the author.

## [PaliGemma-style Vision-Language Model](paligemma_from_scratch.ipynb)

SigLIP vision encoder, Gemma decoder, multimodal projection, image processing, KV cache and inference code.

Requires PyTorch, transformers, Pillow, NumPy, safetensors and fire. Assemble the referenced Python modules, configure image paths and supply compatible model weights/tokenizer. The opening CLIP-style snippet is illustrative pseudocode, and the final cell is a shell launch script.
