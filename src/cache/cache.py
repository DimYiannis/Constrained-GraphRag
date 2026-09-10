"""
    cache for query_pipeline.answer_query()

    need to be disk-backed and not in memory:
    wont survive the cli calls cause its call its a diff process
"""


import hashlib
import pickle
# pickle serializes Python objects into raw bytes
from pathlib import Path

CACHE_FILENAME = "query_cache.pkl"

def _key(query: str, k: int, hops: int, max_new_tokens: int) -> str:
    raw = f"{query}|{k}|{hops}|{max_new_tokens}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def load_cache(cache_dir: Path) -> dict:
    target = cache_dir / CACHE_FILENAME
    if not target.is_file():
        return {}
    try:
        with open(target, "rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError):
        return {}

if __name__ == "__main__":
    print(_key("hello there bro", 2, 2, 500))