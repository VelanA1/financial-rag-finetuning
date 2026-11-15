# Financial RAG Embedding Fine-Tuning

An experiment fine-tuning **BAAI/bge-base-en-v1.5** for financial question-to-document retrieval using a dataset based on NVIDIA SEC filings. Explores **Matryoshka embeddings** at 768, 512, 256, 128, and 64 dimensions with `MultipleNegativesRankingLoss` and `MatryoshkaLoss`.

**Technologies:** Python, PyTorch, Sentence Transformers, Hugging Face Datasets, Transformers, pytrec_eval.

## Notebook
- [`ProEmbeddingFineTuning.ipynb`](ProEmbeddingFineTuning.ipynb) — dataset preparation, baseline evaluation, training, and comparative retrieval evaluation.

## Experiment
- Source dataset: `zeitgeist-ai/financial-rag-nvidia-sec` (Hugging Face).
- 6,300 training / 700 test question pairs as reported in the notebook.
- Four training epochs; model evaluated at multiple embedding dimensions using NDCG@10.

## Recorded results

| Embedding size | Baseline NDCG@10 | After fine-tuning NDCG@10 |
|---|---:|---:|
| 768 | 0.7365 | 0.6995 |

**Fine-tuning did not improve the recorded 768-dimensional retrieval score** (approximately 5.03% lower). The notebook also reports average relative decline across dimensions of 7.38%. This is an experimental finding, not a claim of a better model. Lower-dimensional embeddings illustrate potential storage / representation trade-offs.

## Reproduce
Install packages from `requirements.txt`, then run the notebook in a Python environment with access to Hugging Face downloads. A GPU is strongly recommended. The notebook contains its own package-install cells and writes training datasets/checkpoints locally; these generated files are not provided here. Full training has not been independently rerun for this repository.

## Caveats
The notebook contains experiment scaffolding and intermediate outputs. Downstream model quality and reproducibility may vary with package versions, random splits and hardware. No model weights or third-party dataset files are redistributed here.
