# RAVEN: Long-Horizon Reasoning and Navigation with a Visuo-Spatial-Temporal Memory

<!-- [ :scroll: [`Paper`](https://arxiv.org/)] [ :globe_with_meridians: [`Website`](https://ravenmem.github.io/#)] [ :book: [`BibTeX`](#citing-raven)] -->
[[`Paper`](https://arxiv.org/abs/2606.25206)] [[`Website`](https://ravenmem.github.io/#)] [[`Dataset`](https://huggingface.co/datasets/zzcnewly/RAVEN_QA)] [[`BibTeX`](#bibtex)]

Yixun Hu\*, Zhicheng Zheng\*, Lihan Zha, Chunwei Xing, Rajdeep Singh, Omar Hossain, Antonio Loquercio, and Dhruv Shah (* Equal contribution)

> Note (2026-02-04): If you find any serious bugs or issues, please open an issue and we will address them in a timely manner. Contributions and code improvements are welcome via pull requests. Thank you!

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates
`./.venv` from `pyproject.toml` and the pinned `uv.lock`, downloading CPython
3.10 if your machine does not have it.

```bash
bash docs/raven_setup.sh
source .venv/bin/activate
```

Or, if you already have `uv` installed, equivalently:

```bash
uv sync
source .venv/bin/activate
```

Activating is optional — any command in these docs can instead be prefixed with
`uv run` (e.g. `uv run python raven_qa_run.py ...`), which syncs the environment
on the fly.

Optional extras:

```bash
uv sync --extra gpu                            # faiss-gpu
uv sync --extra flash --no-build-isolation     # flash-attn (needs a CUDA toolkit)
```
### If use open-source VLM models
```bash
curl -fsSL https://ollama.com/download/ollama-{Your-Version}.tgz  |  tar zx -C ~/.cache/ollama/

# for example
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz  |  tar zx -C ~/.cache/ollama/
```

## Running Benchmarks

For step-by-step instructions on running RAVEN, please refer to the following docs:

1. [RAVEN-QA](./docs/README_RAVEN_QA.md): A challenging benchmark for robot long-term memory and navigation with visual information recall, visual reasoning, small object retrieval, and so forth. 

2. [NaVQA](./docs/README_NaVQA.md) A question-answering dataset based on CODa dataset.

3. [FindingDory](./docs/README_FINDINGDORY.md)  A Long-term loco-manipulation dataset and object retrieval benchmark based on Habitat.

# Acknowledgements
- This work is inspired by previous papers or implementations: [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr), [VLFM](https://github.com/bdaiinstitute/vlfm), and [QQMM-embed](https://github.com/QQ-MM/QQMM-embed). We greatly thank these pioneering works.
In particular, this project was originally inspired by and partially built upon the codebase of: https://github.com/NVIDIA-AI-IOT/remembr (ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robots, Abrar Anwar, John Welsh, Joydeep Biswas, Soha Pouya, Yan Chang. [ICRA 2025])

- Datasets: 1. Yadav, K. et al. FindingDory: A Benchmark to Evaluate Memory in Embodied Agents. arXiv:2506.15635, 2025. https://arxiv.org/abs/2506.15635 2. Zhang, A. et al. UT Campus Object Dataset (CODa). Texas Data Repository, Draft Version, 2023. DOI: 10.18738/T8/BBOQMV.



# BibTeX

```
@article{hu2026raven,
  title={RAVEN: Long-Horizon Reasoning \& Navigation with a Visuo-Spatio-Temporal Memory},
  author={Hu, Yixun and Zheng, Zhicheng and Zha, Lihan and Xing, Chunwei and Singh, Rajdeep and Hossain, Omar and Loquercio, Antonio and Shah, Dhruv},
  journal={arXiv preprint arXiv:2606.25206},
  year={2026}
}
```
