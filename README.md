# RLLaMA: Fake News Detection on X Using Contextual Retrieval and LLM Fine-Tuning

**Repository:** `rllama-fake-news-detection`  
**Author:** Arooj Khan (MS in Computer Science, The University of Haripur)  
**Research Status:** Associated with first-author manuscript currently **under peer review**:  
> Arooj Khan, Faten S. Alamri, Tanzila Saba, Fatima Shannag, and Adeel Ahmed. *"Fake News Detection on X Using Generative AI."* (Manuscript under review).

## 1. Research Problem & Motivation

Social media platforms such as X (formerly Twitter) enable rapid information diffusion, making automated misinformation detection critical. Short posts often lack sufficient standalone context for reliable binary classification, while unconstrained large language models (LLMs) risk hallucinating external facts.

**RLLaMA** addresses this challenge through a hybrid, RankRAG-inspired two-stage retrieval and classification pipeline:
1. **Stage 1 — Dense Semantic Retrieval (Bi-Encoder):** Encodes training examples and incoming posts using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional normalized embeddings) and retrieves the top-$N$ ($N=20$) semantically similar labelled training examples via cosine similarity.
2. **Stage 2 — Contextual Reranking (Cross-Encoder):** Reranks the $(query, candidate)$ pairs using `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (max length 256) to select the top-$K$ ($K=4$) most contextually relevant examples.
3. **Stage 3 — Instruction-Tuned LLM Classification (QLoRA SFT):** Constructs a structured retrieval-augmented prompt containing the top-$K$ reranked training examples and fine-tunes a causal LLM (`meta-llama/Llama-3.1-8B-Instruct` or accessible open-weights alternative `Qwen/Qwen2.5-3B-Instruct`) using 4-bit NF4 quantization (`bitsandbytes`) and Low-Rank Adaptation (`PEFT LoRA`, `TRL SFTTrainer`).

## 2. Implementation & Architecture

- **Original Research Notebooks (`notebooks/`):**
  - `notebooks/01_rllama_complete_colab_pipeline.ipynb` — Complete end-to-end Google Colab pipeline (data loading, stratified splitting, dense retrieval, cross-encoder reranking, QLoRA SFT, held-out test set evaluation, regex label parser, and JSON/CSV artifact export).
  - `notebooks/02_fakenewsnet_retrieval_and_sft.ipynb` — Experimental notebook executed on `FakeNewsNet.csv` (`23,196` raw rows; `21,847` deduplicated rows split into `15,291` train / `3,278` validation / `3,278` test rows with `SEED = 42`).
- **Modular Python Package (`src/`):**
  - `src/config.py` — Dataclass and environment-variable configuration (`ExperimentConfig`).
  - `src/data_utils.py` — Label normalization (`Fake=0, Real=1`), text cleaning, deduplication, and 70/15/15 stratified splitting (`SEED = 42`).
  - `src/retrieval.py` — `ContextualRankRAGRetriever` implementing bi-encoder indexing and cross-encoder reranking (with self-exclusion `exclude_index` during training prompt construction to prevent label leakage).
  - `src/prompts.py` — Structured prompt builder (`build_prompt`) and regex output parser (`parse_label`).
  - `src/train_and_eval.py` — CLI entry point for 4-bit NF4 QLoRA training and held-out test set evaluation.

## 3. Dataset & Experimental Setup

- **Primary Dataset (`FakeNewsNet.csv`):**
  - **Raw Shape:** `23,196` rows, `5` columns (`['title', 'news_url', 'source_domain', 'tweet_num', 'real']`)
  - **Raw Class Distribution:** `17,441` Real (`1`), `5,755` Fake (`0`)
  - **After Deduplication & Cleaning:** `21,847` unique `(text, label)` pairs
  - **Stratified Splits (`SEED = 42`):** `15,291` Train / `3,278` Validation / `3,278` Held-Out Test
- **Reported MS Thesis Result:** Achieved **97% classification accuracy** in distinguishing fake and real news posts on X in the MS thesis evaluation setting.

## 4. Installation & Execution

```bash
git clone https://github.com/Arooj-khan301/rllama-fake-news-detection.git
cd rllama-fake-news-detection
pip install -r requirements.txt

# Run unit tests
PYTHONPATH=. python3 -m unittest discover -s tests -v

# Run training and evaluation (requires CUDA GPU for 4-bit QLoRA)
python3 -m src.train_and_eval --csv-path data/FakeNewsNet.csv --text-col title --label-col real
```
