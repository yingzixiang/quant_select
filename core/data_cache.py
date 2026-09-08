import pickle
import hashlib
import os
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    raw = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(key: str, max_age_seconds: int = 86400) -> object:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > max_age_seconds:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_set(key: str, data: object):
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def cached_api_call(func, *args, max_age_seconds=86400, **kwargs):
    key = cache_key(func.__name__, args, kwargs)
    cached = cache_get(key, max_age_seconds)
    if cached is not None:
        return cached
    result = func(*args, **kwargs)
    cache_set(key, result)
    return result


def clear_cache(max_age_seconds: float = None):
    """清除过期缓存，max_age_seconds=None 则清除所有"""
    for f in CACHE_DIR.glob("*.pkl"):
        if max_age_seconds is None or (time.time() - f.stat().st_mtime) > max_age_seconds:
            f.unlink()
