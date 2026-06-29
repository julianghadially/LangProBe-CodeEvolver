"""Retrieval database for RAGQAArenaTech.

The LoTTE 'technology' corpus plus a precomputed OpenAI-embedding index, loaded
once per process and shared across all program instances. This is the *database*
(passages only, no gold labels) -- safe for the optimizer/architect to read.

It is intentionally decoupled from the program so that:
  - the program owns no IO and can be freely evolved by the optimizer, and
  - the 3.9GB index is loaded a single time and shared by every program that uses
    it (the default program, the benchmark instance, and any new program the
    optimizer writes), instead of once per instance.

Thread-safety: the retriever is a shared, read-only singleton with no mutable
per-call state. dspy ``Evaluate`` parallelizes with a thread pool (one process),
calling the same program -- so the index is shared across threads, never copied.
``search()`` only reads ``self.index``/``self.corpus`` (torch matmul/topk and the
litellm embed call are per-call and non-mutating), so concurrent calls are safe.
"""

import os
import threading
from pathlib import Path

import requests
import torch
import ujson
from litellm import embedding as Embed

DATA_DIR = "langProBe/RAGQAArenaTech/data"
CORPUS_URL = "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl"
INDEX_URL = "https://huggingface.co/dspy/cache/resolve/main/index.pt"


def _ensure_file(url: str) -> str:
    """Return the local path for ``url``'s file, downloading only if it is absent.

    No network call is made when the file already exists -- the common case, since
    the database is present locally or mounted into the CodeEvolver sandbox. When a
    download is needed it streams to a ``.part`` file and renames atomically, so a
    file that exists is always complete (no partial-download / size-check dance).
    """
    path = os.path.join(DATA_DIR, os.path.basename(url))
    if os.path.exists(path):
        return path
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{os.path.basename(url)}'...")
    tmp = path + ".part"
    with requests.get(url, stream=True) as r, open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    os.rename(tmp, path)
    print(f"Downloaded '{os.path.basename(url)}'")
    return path


class EmbeddingRetriever:
    """Dense top-k retrieval over the LoTTE 'technology' corpus.

    Embeds queries with OpenAI ``text-embedding-3-small`` and scores them against a
    precomputed passage-embedding index via dot product.
    """

    def __init__(self, max_characters: int = 4000):
        self.max_characters = max_characters
        corpus_path = _ensure_file(CORPUS_URL)
        index_path = _ensure_file(INDEX_URL)
        with open(corpus_path) as f:
            self.corpus = [ujson.loads(line) for line in f]
        self.index = torch.load(index_path, weights_only=True)

    def search(self, query: str, k: int = 5):
        query_embedding = torch.tensor(
            Embed(input=query, model="text-embedding-3-small").data[0]["embedding"]
        )
        topk_scores, topk_indices = torch.matmul(self.index, query_embedding).topk(k)
        topk = [
            dict(score=score.item(), **self.corpus[idx])
            for idx, score in zip(topk_indices, topk_scores)
        ]
        return [doc["text"][: self.max_characters] for doc in topk]

    # The corpus + 3.9GB index are immutable, read-only infrastructure -- not
    # optimizable parameters -- so copies must SHARE them rather than clone 4GB.
    # This keeps the index single-copy even if an optimizer deepcopies the program.
    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self


_default_retriever = None
_retriever_lock = threading.Lock()


def get_default_retriever() -> EmbeddingRetriever:
    """Lazily build and cache the default retriever (one index load per process).

    Double-checked locking so concurrent first-calls from worker threads build
    exactly one retriever (a single 3.9GB index load), not one per racing thread.
    """
    global _default_retriever
    if _default_retriever is None:
        with _retriever_lock:
            if _default_retriever is None:
                _default_retriever = EmbeddingRetriever()
    return _default_retriever
