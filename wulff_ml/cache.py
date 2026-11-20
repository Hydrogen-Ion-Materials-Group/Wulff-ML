"""
Generic hash-based caching layer for the Wulff-ML pipeline.

Wraps any function with `@cached(...)`. On every call it:
  1. Hashes the function name + all arguments (DataFrames, Series, arrays,
     dicts, nested structures, file paths, config dicts — all handled).
  2. If a cache entry with that exact hash already exists on disk, loads
     and returns it instantly — the wrapped function body never runs.
  3. Otherwise runs the function, pickles the *exact* return value (so
     re-loading is lossless regardless of type), and additionally writes
     out human-readable copies of every piece of the result (.csv for
     DataFrames/Series, .npy for arrays, .json for plain dicts/lists,
     .pkl for anything else such as fitted models) so you can inspect or
     reuse results outside Python without re-running anything.

Cache invalidation is automatic: change the input data, change a config
parameter, change a function argument -> the hash changes -> it recomputes.
Nothing else changes -> instant cache hit.

Layout:
    cache/
      manifest.json                 <- master index: task -> key -> metadata
      <task_name>/
        <hash>.pkl                  <- exact pickled return value (joblib)
        <hash>__exports/            <- human-readable breakdown of the result
          out_0.csv / out_1.npy / ...

Force a clean recompute for one call:
    train_xgb(splits, _force_recompute=True)

Force a clean recompute for everything in a run (set before running scripts):
    export WULFF_CACHE_FORCE=1
"""
import functools
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from wulff_ml import config

CACHE_DIR = os.path.join(config.PROJECT_ROOT, "cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")


# ==============================================================================
# Manifest IO
# ==============================================================================
def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(manifest: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp_path, MANIFEST_PATH)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


# ==============================================================================
# Stable hashing of arbitrary Python / pandas / numpy objects
# ==============================================================================
def _hash_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_into(obj, h: "hashlib._Hash"):
    if isinstance(obj, pd.DataFrame):
        h.update(b"DF")
        h.update(",".join(map(str, obj.columns)).encode())
        try:
            h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
        except Exception:
            h.update(obj.to_csv(index=True).encode())
    elif isinstance(obj, pd.Series):
        h.update(b"SR")
        h.update(str(obj.name).encode())
        try:
            h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
        except Exception:
            h.update(obj.to_csv(index=True).encode())
    elif isinstance(obj, pd.Index):
        h.update(b"IDX")
        h.update(",".join(map(str, obj.tolist())).encode())
    elif isinstance(obj, np.ndarray):
        h.update(b"NP")
        h.update(str(obj.dtype).encode())
        h.update(obj.tobytes())
    elif isinstance(obj, dict):
        h.update(b"DC")
        for k in sorted(obj.keys(), key=str):
            h.update(str(k).encode())
            _hash_into(obj[k], h)
    elif isinstance(obj, (list, tuple)):
        h.update(b"LS")
        for item in obj:
            _hash_into(item, h)
    elif isinstance(obj, str) and os.path.isfile(obj):
        # Looks like a path to an existing file -> hash its *contents*, so
        # editing/regenerating an upstream CSV correctly invalidates the cache.
        h.update(b"FILE")
        h.update(_hash_file(obj).encode())
    elif isinstance(obj, (int, float, bool, str, type(None))):
        h.update(b"PR")
        h.update(repr(obj).encode())
    else:
        # Fitted models, custom objects, etc. Best-effort stable-ish hash.
        try:
            h.update(json.dumps(obj, sort_keys=True, default=str).encode())
        except Exception:
            h.update(repr(obj).encode())


def stable_hash(*args, **kwargs) -> str:
    h = hashlib.sha256()
    _hash_into(args, h)
    _hash_into(kwargs, h)
    return h.hexdigest()


# ==============================================================================
# Human-readable export (best-effort, recursive)
# ==============================================================================
def _is_shap_explanation(obj) -> bool:
    return hasattr(obj, "values") and hasattr(obj, "base_values") and hasattr(obj, "data")


def _export_one(obj, base_path: str):
    try:
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(base_path + ".csv", index=True)
        elif isinstance(obj, pd.Series):
            obj.to_frame(name=obj.name or "value").to_csv(base_path + ".csv", index=True)
        elif isinstance(obj, np.ndarray):
            np.save(base_path + ".npy", obj)
        elif isinstance(obj, dict):
            os.makedirs(base_path, exist_ok=True)
            for k, v in obj.items():
                _export_one(v, os.path.join(base_path, _safe_name(str(k))))
        elif isinstance(obj, (list, tuple)):
            os.makedirs(base_path, exist_ok=True)
            for i, v in enumerate(obj):
                _export_one(v, os.path.join(base_path, f"item_{i}"))
        elif isinstance(obj, (int, float, bool, str, type(None))):
            with open(base_path + ".json", "w") as f:
                json.dump(obj, f, indent=2, default=str)
        elif _is_shap_explanation(obj):
            np.save(base_path + "__values.npy", np.asarray(obj.values))
            try:
                np.save(base_path + "__base_values.npy", np.asarray(obj.base_values))
            except Exception:
                pass
            try:
                with open(base_path + "__feature_names.json", "w") as f:
                    json.dump(list(obj.feature_names), f, indent=2)
            except Exception:
                pass
        else:
            # Anything reaching here is a non-primitive object the checks above
            # didn't recognize (a fitted model, a custom class instance, etc.) —
            # always pickle it. (json.dumps(..., default=str) would "succeed"
            # on virtually anything by silently stringifying it, which produces
            # a useless export, so we don't attempt JSON for this branch.)
            joblib.dump(obj, base_path + ".pkl")
    except Exception as e:
        # Exports are a convenience, never let them break the actual pipeline.
        print(f"  [cache] WARN: export failed for {base_path}: {e}")


def _export_readable(result, base_path: str):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    _export_one(result, base_path)


# ==============================================================================
# The decorator
# ==============================================================================
def cached(task_name: str = None, ignore_kwargs=None, export: bool = True):
    """
    Decorator: cache a function's return value, keyed by a hash of its
    function name + all positional/keyword arguments.

    Args:
        task_name:     cache namespace (defaults to the function's qualname).
        ignore_kwargs: kwarg names that affect only logging/verbosity, not
                       the actual result (e.g. "verbose") — excluded from
                       the hash so identical computations with different
                       verbosity still hit the same cache entry.
        export:        also dump human-readable .csv/.json/.npy copies of
                        the result (default True).
    """
    ignore_kwargs = set(ignore_kwargs or [])

    def decorator(fn):
        name = task_name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            force = kwargs.pop("_force_recompute", False) or os.environ.get("WULFF_CACHE_FORCE") == "1"
            key_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_kwargs}
            key = stable_hash(*args, **key_kwargs)
            short_key = key[:16]

            task_dir = os.path.join(CACHE_DIR, _safe_name(name))
            cache_path = os.path.join(task_dir, f"{short_key}.pkl")

            manifest = _load_manifest()
            entry = manifest.get(name, {}).get(short_key)

            if not force and entry and os.path.exists(cache_path):
                print(f"  [cache] HIT  '{name}' (key={short_key}) — skipping computation, loading saved result")
                return joblib.load(cache_path)

            print(f"  [cache] MISS '{name}' (key={short_key}) — running and caching")
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0

            # Some tasks report a failure as {"error": "..."} instead of raising
            # (e.g. a Materials Project download that timed out). Never store
            # those: otherwise one network hiccup would be replayed forever.
            if isinstance(result, dict) and set(result) == {"error"}:
                print(f"  [cache] '{name}' failed ({result['error']}) — result not cached, will retry next time")
                return result

            os.makedirs(task_dir, exist_ok=True)
            joblib.dump(result, cache_path)

            if export:
                export_base = os.path.join(task_dir, f"{short_key}__export")
                _export_readable(result, export_base)

            manifest.setdefault(name, {})[short_key] = {
                "path": cache_path,
                "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 3),
            }
            _save_manifest(manifest)
            return result

        wrapper.__wrapped_cached__ = fn
        wrapper.__cache_task_name__ = task_name
        return wrapper

    return decorator


# ==============================================================================
# Maintenance helpers
# ==============================================================================
def clear(task_name: str = None):
    """Delete cached results. With no argument, wipes the entire cache."""
    import shutil

    manifest = _load_manifest()
    if task_name is None:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        print("  [cache] cleared ALL cached results")
        return
    task_dir = os.path.join(CACHE_DIR, _safe_name(task_name))
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    manifest.pop(task_name, None)
    _save_manifest(manifest)
    print(f"  [cache] cleared cached results for '{task_name}'")


def status():
    """Print a summary of everything currently cached."""
    manifest = _load_manifest()
    if not manifest:
        print("Cache is empty.")
        return
    total = 0
    for task, entries in sorted(manifest.items()):
        print(f"{task}  ({len(entries)} cached result(s))")
        for key, meta in entries.items():
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("timestamp", 0)))
            print(f"    {key}  saved={ts}  took={meta.get('elapsed_seconds', '?')}s")
            total += 1
    print(f"\nTotal cached results: {total}")
