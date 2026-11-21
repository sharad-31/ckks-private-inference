"""
CKKS scheme context and preset parameters for TenSEAL / Microsoft SEAL.
Supports standard, precise, deep, and deeper configurations as evaluated in the project report.
"""
import tenseal as ts

# Scheme presets matching the study's depth and precision requirements
PRESETS = {
    # standard: 3 levels (conv, square, linear readout) - N=8192, ~1e-3 logit error
    "standard": {
        "poly": 8192,
        "coeff": [50, 30, 30, 30, 50],
        "scale": 2**30,
        "depth": 3,
        "description": "N=8192, scale 2^30; fast inference, ~1e-3 logit error"
    },
    # precise: 4 levels (higher coefficient headroom) - N=16384, ~1e-5 logit error
    "precise": {
        "poly": 16384,
        "coeff": [60, 40, 40, 40, 40, 60],
        "scale": 2**40,
        "depth": 4,
        "description": "N=16384, scale 2^40; high precision, 0 test flips across all CNN models"
    },
    # deep: 7 levels - allows encrypted threshold decision (degree-3 sign polynomial, 2 iterations)
    "deep": {
        "poly": 16384,
        "coeff": [60] + [40] * 7 + [60],
        "scale": 2**40,
        "depth": 7,
        "sign_iters": 2,
        "description": "N=16384, 7 multiplicative levels; encrypted sign polynomial threshold"
    },
    # deeper: 9 levels - allows encrypted threshold decision (degree-3 sign polynomial, 3 iterations)
    "deeper": {
        "poly": 32768,
        "coeff": [60] + [40] * 9 + [60],
        "scale": 2**40,
        "depth": 9,
        "sign_iters": 3,
        "description": "N=32768, 9 multiplicative levels; higher iteration sign polynomial"
    }
}


def make_ckks_context(preset="precise", galois=True):
    """
    Constructs a TenSEAL CKKS scheme context using the configured preset.

    Args:
        preset (str): One of 'standard', 'precise', 'deep', 'deeper'.
        galois (bool): Whether to generate Galois keys (required for rotations/im2col conv).
    Returns:
        ts.Context: Initialized TenSEAL CKKS context with global scale configured.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
    p = PRESETS[preset]
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=p["poly"],
        coeff_mod_bit_sizes=p["coeff"]
    )
    ctx.global_scale = p["scale"]
    if galois:
        ctx.generate_galois_keys()
    return ctx


def get_preset_slots(preset):
    """Returns number of SIMD plaintext slots available for a preset (poly / 2)."""
    return PRESETS[preset]["poly"] // 2


def get_preset_levels(preset):
    """Returns number of multiplicative levels available before ciphertext is exhausted."""
    return len(PRESETS[preset]["coeff"]) - 2
