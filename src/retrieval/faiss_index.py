from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


class NumpyFlatIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = l2_normalize(np.asarray(queries, dtype=np.float32))
        similarities = query_vectors @ self.vectors.T
        top_k = min(top_k, self.vectors.shape[0])
        indices = np.argpartition(-similarities, kth=top_k - 1, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        sorted_indices = np.take_along_axis(indices, order, axis=1)
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores, sorted_indices


def build_index(
    vectors: np.ndarray,
    metric: str = "cosine",
    use_gpu: bool = False,
) -> Any:
    if metric != "cosine":
        raise ValueError("Only cosine metric is supported")

    vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    if faiss is None:
        return NumpyFlatIndex(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    if use_gpu:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    index.add(vectors)
    return index


def search_index(index: Any, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    query_vectors = l2_normalize(np.asarray(queries, dtype=np.float32))
    scores, indices = index.search(query_vectors, top_k)
    return scores, indices


def save_index(index: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if faiss is not None and hasattr(index, "ntotal"):
        index_to_save = faiss.index_gpu_to_cpu(index) if "Gpu" in type(index).__name__ else index
        faiss.write_index(index_to_save, str(output_path))
        return output_path

    with output_path.open("wb") as file:
        np.save(file, index.vectors)
    return output_path


def load_index(index_path: str | Path) -> Any:
    index_path = Path(index_path)
    if faiss is not None:
        return faiss.read_index(str(index_path))

    with index_path.open("rb") as file:
        vectors = np.load(file)
    return NumpyFlatIndex(vectors)


__all__ = [
    "NumpyFlatIndex",
    "build_index",
    "l2_normalize",
    "load_index",
    "save_index",
    "search_index",
]
