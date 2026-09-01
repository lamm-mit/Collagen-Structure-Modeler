#!/usr/bin/env python3
"""
prices.py — dated hardware price table, and the resource-seconds -> dollars map.

Dollars are always derived, never measured. Everything upstream of this file
records *seconds*; this file is the only place a currency amount is introduced.
When Modal reprices, edit PRICES_USD_PER_SECOND and re-derive — the measured
seconds stay valid, and every published figure can be regenerated.

Source: https://modal.com/pricing, fetched 2026-08-12.
"""

PRICE_DATE = "2026-08-12"
PRICE_SOURCE = "https://modal.com/pricing"

# USD per second.
PRICES_USD_PER_SECOND = {
    # GPU, billed per second of container lifetime (includes cold start).
    "gpu:B200":        0.001736,
    "gpu:H200":        0.001261,
    "gpu:H100":        0.001097,
    "gpu:A100-80GB":   0.000694,
    "gpu:A100-40GB":   0.000583,
    "gpu:L40S":        0.000542,
    "gpu:A10":         0.000306,
    "gpu:L4":          0.000222,
    "gpu:T4":          0.000164,
    # Non-GPU resources, billed alongside.
    "cpu:core":        0.0000131,   # minimum 0.125 cores per container
    "mem:gib":         0.00000222,
}

# Minimum VRAM in GiB per tier — used to justify "the cheapest tier that fits",
# not to predict success. T4 is Turing (sm_75) and has no bf16, so it cannot run
# Chai-1 or Protenix regardless of capacity; that exclusion is architectural.
TIER_VRAM_GIB = {
    "gpu:T4": 16, "gpu:L4": 24, "gpu:A10": 24, "gpu:L40S": 48,
    "gpu:A100-40GB": 40, "gpu:A100-80GB": 80, "gpu:H100": 80,
}
TIER_SUPPORTS_BF16 = {
    "gpu:T4": False,       # Turing sm_75
    "gpu:L4": True,        # Ada sm_89
    "gpu:A10": True,       # Ampere sm_86
    "gpu:L40S": True, "gpu:A100-40GB": True, "gpu:A100-80GB": True, "gpu:H100": True,
}


def cost_usd(seconds: float, resource: str, n: float = 1.0) -> float:
    """Cost of `n` units of `resource` held for `seconds`.

    >>> round(cost_usd(60, "gpu:L40S"), 6)      # one L40S-minute
    0.03252
    >>> round(cost_usd(60, "cpu:core", n=1), 6)  # one CPU-core-minute
    0.000786
    """
    if resource not in PRICES_USD_PER_SECOND:
        raise ValueError(f"unknown resource {resource!r}; "
                         f"expected one of {sorted(PRICES_USD_PER_SECOND)}")
    return seconds * n * PRICES_USD_PER_SECOND[resource]


def container_cost_usd(seconds: float, gpu: str = None,
                       cpu_cores: float = 0.125, mem_gib: float = 0.125) -> float:
    """Total cost of a container held for `seconds`.

    Modal bills GPU, CPU and memory together for the container's whole lifetime,
    so a GPU row is never GPU-seconds alone. Defaults are Modal's minimum
    request (0.125 cores); the co-folder apps set neither cpu= nor memory=, so
    the minimum is what they are billed for.
    """
    total = cost_usd(seconds, "cpu:core", cpu_cores) + cost_usd(seconds, "mem:gib", mem_gib)
    if gpu:
        total += cost_usd(seconds, gpu)
    return total


def mb_to_gib(mb: float) -> float:
    """Decimal MB (as NVML reports, bytes/1e6) -> binary GiB (as VRAM is sold).

    Dividing MB by 1024 is a common slip and overstates capacity by ~7%, which
    can push a device over a tier boundary that it actually fits under.

    >>> round(mb_to_gib(15047.7), 2)     # Chai-1 peak: 14.01 GiB, not 14.69
    14.01
    """
    return mb * 1e6 / (1 << 30)


def cheapest_tier_for(peak_vram_gib: float, needs_bf16: bool = True,
                      headroom: float = 1.25) -> str:
    """The cheapest tier whose VRAM fits `peak_vram_gib` with headroom.

    Reports what the measurement implies about deployment cost, rather than
    requiring a tier sweep to discover it empirically.
    """
    need = peak_vram_gib * headroom
    fits = [t for t, v in TIER_VRAM_GIB.items()
            if v >= need and (TIER_SUPPORTS_BF16[t] or not needs_bf16)]
    return min(fits, key=lambda t: PRICES_USD_PER_SECOND[t]) if fits else None


if __name__ == "__main__":
    print(f"Modal prices as of {PRICE_DATE}  ({PRICE_SOURCE})\n")
    print(f"{'resource':18s} {'USD/s':>12s} {'USD/hour':>10s}")
    print("-" * 42)
    for r, p in PRICES_USD_PER_SECOND.items():
        print(f"{r:18s} {p:12.7f} {p * 3600:10.4f}")
