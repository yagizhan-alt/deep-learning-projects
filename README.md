# Deep Learning Projects

A learning portfolio by **Yağızhan Dağ**, exploring deep learning models and GPU programming through 22 PyTorch and Triton notebooks.

The collection covers representation learning, language and vision models, parameter-efficient fine-tuning, and attention kernels. It documents implementation practice and work in progress.

## Explore the projects

| Area | Notebooks | Topics |
| --- | ---: | --- |
| [Autoencoders](autoencoders/) | 4 | AE, VAE, VQ-VAE, residual vector quantization |
| [Language models](language-models/) | 3 | Masked language modeling, RoBERTa pretraining and Llama 2 |
| [Computer vision](vision/) | 1 | Vision Transformer and ImageNet training code |
| [Vision-language models](vision-language/) | 2 | Llama 4-style components, PaliGemma, multimodal projection and MoE |
| [Parameter-efficient fine-tuning](parameter-efficient-finetuning/) | 2 | LoRA adapters, model wrapping and MNIST parameterization |
| [GPU kernels](gpu-kernels/) | 10 | FlashAttention, Triton matmul, softmax, LayerNorm and embeddings |

## Reading and running

Browse any notebook directly on GitHub, or clone the repository and open it in JupyterLab:

```bash
git clone https://github.com/yagizhan-alt/deep-learning-projects.git
cd deep-learning-projects
python -m pip install jupyterlab
jupyter lab
```

Each folder's README lists the relevant dependencies and environment requirements. Install the project-specific packages in an isolated environment. Dependency versions have not yet been pinned or tested together. Triton notebooks require a compatible GPU environment; large model configurations and ImageNet training need additional resources and data.

## Current status

This first portfolio import preserves the uploaded code and markdown. Execution outputs, counts and transient notebook metadata were cleared for readable version control. No model training, GPU tests or benchmarks were run during the import.

Some notebooks contain unfinished cells, syntax errors, missing imports or references to external modules. A successful syntax parse does not establish runtime correctness. See [VALIDATION.md](VALIDATION.md) for the static-review findings and known blockers.

## Sources and attribution

The notebooks are educational implementation studies. Original upload names are preserved in [SOURCES.md](SOURCES.md); tutorial-style titles are not evidence of original research or an independent reproduction result. Exact upstream tutorial/repository links and a description of personal modifications are pending author confirmation. Existing code comments and markdown are retained. No new license is assigned to code of unverified provenance.
