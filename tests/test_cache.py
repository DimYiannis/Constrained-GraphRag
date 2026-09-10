# uv run pytest tests/test_cache.py -v
from src.cache import cache


def test_key_insert_then_store_result(tmp_path):
    c = cache.load_cache(tmp_path)
    key = cache._key("hello there bro", 2, 2, 500)
    c[key] = {"answer": "hi", "sources": []}
    cache.save_cache(c, tmp_path)

    reloaded = cache.load_cache(tmp_path)
    assert reloaded == c

    cache.store_result(c, "heelllo", 2, 2, 300, {"answer": "hi", "sources": []})
    assert cache.get_cached_result(c, "heelllo", 2, 2, 300) == {
        "answer": "hi", "sources": []
    }
