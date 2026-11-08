"""Mulberry32, Box–Muller, and Knuth Poisson. Bit-matched to the reference simulator."""

from __future__ import annotations


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    if x >= 0x80000000:
        return x - 0x100000000
    return x


def _imul(a: int, b: int) -> int:
    return _i32(_i32(a) * _i32(b))


def mulberry32(seed: int):
    """Uniform draws in [0, 1). Same 32-bit recurrence as the reference."""
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = _i32(a + 0x6D2B79F5)
        ua = a & 0xFFFFFFFF
        t = _imul(a ^ (ua >> 15), 1 | a)
        ut = t & 0xFFFFFFFF
        t = _i32(t + _imul(t ^ (ut >> 7), 61 | t)) ^ t
        t = _i32(t)
        ut = t & 0xFFFFFFFF
        return ((t ^ (ut >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rng


def gauss(rng) -> float:
    import math

    u = rng()
    v = rng()
    if u < 1e-12:
        u = 1e-12
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def _js_round(x: float) -> int:
    """Math.round: halfway cases go toward +∞."""
    import math

    return int(math.floor(x + 0.5))


def poisson(rng, lambda_: float) -> int:
    import math

    if lambda_ <= 0:
        return 0
    if lambda_ > 40:
        z = gauss(rng)
        return max(0, _js_round(lambda_ + math.sqrt(lambda_) * z))
    L = math.exp(-lambda_)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng()
        if p <= L:
            break
    return k - 1


def clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
