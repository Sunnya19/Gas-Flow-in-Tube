import math


def compute_knudsen_number(
    mean_free_path: float,
    characteristic_length: float,
) -> float:
    """
    Compute the Knudsen number.

    Kn = lambda / L
    """
    if characteristic_length == 0.0:
        return float("nan")
    return mean_free_path / characteristic_length


def classify_knudsen_number(kn: float) -> str:
    """
    Classify the flow regime based on the Knudsen number.
    """
    if not math.isfinite(kn):
        return "unknown"
    if kn < 0.01:
        return "continuum"
    if kn < 0.1:
        return "slip"
    if kn < 10.0:
        return "transition"
    return "free_molecular"
