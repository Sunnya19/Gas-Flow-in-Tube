import math

from src.measurements.knudsen import (
    classify_knudsen_number,
    compute_knudsen_number,
)


def test_compute_knudsen_number():
    assert compute_knudsen_number(5.0, 10.0) == 0.5
    assert math.isnan(compute_knudsen_number(1.0, 0.0))


def test_classify_knudsen_number_continuum():
    assert classify_knudsen_number(0.001) == "continuum"


def test_classify_knudsen_number_slip():
    assert classify_knudsen_number(0.05) == "slip"


def test_classify_knudsen_number_transition():
    assert classify_knudsen_number(1.0) == "transition"


def test_classify_knudsen_number_free_molecular():
    assert classify_knudsen_number(20.0) == "free_molecular"
