"""Two-stage dense retrieval and cross-encoder contextual reranking (RankRAG-inspired).

Note: Fixes the typo in the original `Untitled15.ipynb` script where `ereranker`
was initialized instead of `reranker`.
"""

from typing import Dict, List, Optional
import numpy as np


class ContextualRankRAGRetriever:
    """Bi-encoder dense retrieval followed by cross-encoder semantic reranking."""

    def __init__(
        self,
        embed_model_name: str,
        reranker_name: str,
        top_candidates: int = 20,
        top_k: int = 4,
    ) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(reranker_name, max_length=256)
        self.top_candidates = top_candidates
        self.top_k = top_k
        self.train_texts: List[str] = []
        self.train_labels: List[int] = []
        self.train_embeddings: Optional[np.ndarray] = None

    def fit_index(
        self, train_texts: List[str], train_labels: List[int], batch_size: int = 32
    ) -> np.ndarray:
        """Encodes training corpus into normalized dense embeddings."""
        self.train_texts = list(train_texts)
        self.train_labels = [int(x) for x in train_labels]
        self.train_embeddings = self.embedder.encode(
            self.train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self.train_embeddings

    def retrieve_and_rerank(
        self,
        query: str,
        top_candidates: Optional[int] = None,
        top_k: Optional[int] = None,
        exclude_index: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Retrieves top-N dense candidates and reranks to top-K via CrossEncoder."""
        if self.train_embeddings is None:
            raise RuntimeError("Call fit_index() before retrieve_and_rerank().")
        n_cand = top_candidates or self.top_candidates
        k = top_k or self.top_k

        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        similarities = self.train_embeddings @ q_emb
        if exclude_index is not None and 0 <= exclude_index < len(similarities):
            similarities[exclude_index] = -1e9

        candidate_ids = np.argsort(similarities)[::-1][
            : min(n_cand, len(self.train_texts))
        ]
        if len(candidate_ids) == 0:
            return []

        candidate_texts = [self.train_texts[int(i)] for i in candidate_ids]
        pair_scores = self.reranker.predict(
            [(query, candidate) for candidate in candidate_texts],
            show_progress_bar=False,
        )
        order = np.argsort(pair_scores)[::-1][: min(k, len(candidate_ids))]

        results: List[Dict[str, object]] = []
        for rank, j in enumerate(order, start=1):
            idx = int(candidate_ids[j])
            results.append({
                "rank": rank,
                "text": self.train_texts[idx],
                "label": int(self.train_labels[idx]),
                "retrieval_similarity": float(similarities[idx]),
                "rerank_score": float(pair_scores[j]),
            })
        return results
