"""
General utility functions: chunked multiprocessing execution, random seeding,
and numerical helper routines.
"""
import os
import random
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def set_seed(seed=42):
    """Sets deterministic random seed for Python, NumPy, and PyTorch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def next_pow2(n):
    """Returns the smallest power of 2 greater than or equal to n."""
    n = int(n)
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def run_chunked_inference(infer_fn, items, chunk_size, cache_dir, workers=4, init_fn=None, init_args=()):
    """
    Executes inference over a dataset in parallel chunks, writing each chunk to disk.
    If execution was previously interrupted, finished chunks in cache_dir are reused.

    Args:
        infer_fn: Worker inference callable accepting one item.
        items: List or sequence of input tuples/items.
        chunk_size (int): Number of items per chunk file.
        cache_dir (str or Path): Directory where chunk .npy files are stored.
        workers (int): Number of parallel worker processes.
        init_fn: Optional process initializer.
        init_args: Arguments passed to init_fn.

    Returns:
        np.ndarray: Concatenated array of all decrypted model predictions.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    n_items = len(items)
    n_chunks = -(-n_items // chunk_size)  # ceiling division

    t0 = time.time()
    with Pool(processes=workers, initializer=init_fn, initargs=init_args) as pool:
        for ci, start_idx in enumerate(range(0, n_items, chunk_size)):
            chunk_file = cache_path / f"chunk_{ci:05d}.npy"
            if chunk_file.exists():
                continue
            
            end_idx = min(start_idx + chunk_size, n_items)
            chunk_items = items[start_idx:end_idx]
            
            # Map inference over chunk
            results = pool.map(infer_fn, chunk_items, chunksize=min(16, max(1, len(chunk_items) // (workers * 2))))
            np.save(chunk_file, np.asarray(results))
            
            elapsed = time.time() - t0
            print(f"  Processed {end_idx}/{n_items} items | {elapsed/60:.2f} min elapsed", flush=True)

    # Gather and concatenate all chunks in order
    all_chunks = []
    for ci in range(n_chunks):
        chunk_file = cache_path / f"chunk_{ci:05d}.npy"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing expected chunk file: {chunk_file}")
        all_chunks.append(np.load(chunk_file))

    return np.concatenate(all_chunks, axis=0)
