import os
import math
import json
import time
import hashlib
from typing import List

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def sha_to_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def small_primes(n: int) -> List[int]:
    """Return first n primes using simple sieve."""
    if n <= 0:
        return []
    if n < 6:
        bound = 15
    else:
        bound = int(n * (math.log(n) + math.log(math.log(n))) * 1.2 + 10)
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(bound ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start: bound + 1: step] = [False] * (((bound - start) // step) + 1)
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    if len(primes) >= n:
        return primes[:n]
    x = bound + 1
    while len(primes) < n:
        is_p = True
        r = int(x ** 0.5) + 1
        for p in primes:
            if p > r:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 1
    return primes[:n]

def normalize(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)
