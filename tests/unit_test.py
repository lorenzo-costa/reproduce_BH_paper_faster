import pytest
from ..src.dgps import NormalGenerator
from ..src.methods import BonferroniHochberg, FalseDiscoveryRate, Bonferroni
import numpy as np


def test_normal_data_generator():
    """Test correct sample generation"""
    loc = 0
    scale = 1
    rng = np.random.default_rng(42)
    gen = NormalGenerator(loc=loc, scale=scale)
    data = gen.generate(10000, rng=rng)
    assert data.mean() == pytest.approx(loc, abs=0.1)
    assert data.std() == pytest.approx(scale, abs=0.1)
    assert gen.name == f"Normal(mu={loc}, sigma={scale})"

    rng2 = np.random.default_rng(42)
    data2 = gen.generate(10000, rng=rng2)
    assert np.array_equal(data, data2)

@pytest.mark.parametrize("method_class, expected_name", [
    (Bonferroni, "Bonferroni Correction"),
    (BonferroniHochberg, "Bonferroni-Hochberg Correction"),
    (FalseDiscoveryRate, "FDR Correction"),
])
def test_method_names(method_class, expected_name):
    method = method_class()
    assert method.name == expected_name

# @pytest.mark.parametrize("pvals, alpha, expected", [
#     (np.array([0.01, 0.04, 0.03, 0.20]), 0.05, np.array([ True, False, False, False])),
#     (np.array([0.10, 0.20, 0.30]), 0.05, np.array([False, False, False])),
#     (np.array([0.001, 0.002, 0.003]), 0.01, np.array([True, True, True])),
# ])
# def test_bonferroni_correction():
#     pass

@pytest.mark.parametrize("pvals, alpha, expected", [
    ([0.01], 0.05, [True]),
    ([0.2], 0.05, [False]),
    ([0.04, 0.5], 0.05, [False, False]),
    ([0.06, 0.07, 0.2, 0.5], 0.05, [False, False, False, False]),
    ([0.0001, 0.001, 0.002, 0.003], 0.05, [True, True, True, True]),
    ([0.0005, 0.02, 0.04, 0.06, 0.2], 0.05, [True, False, False, False, False]),
    ([0.025, 0.025, 0.06, 0.06], 0.05, [False, False, False, False]),
    ([0.2, 0.15, 0.04, 0.02, 0.001], 0.05, [False, False, False, False, True]),
    ([0.05, 0.05, 0.05, 0.05], 0.05, [True, True, True, True]),
    ([0.001, 0.02, 0.025, 0.03, 0.2], 0.05, [True, False, False, False, False]),
    ([0.009, 0.5, 0.6, 0.7, 0.8], 0.05, [True, False, False, False, False])
])
def test_bonferroni_hochberg_correction(pvals, alpha, expected):
    bonf_hoch = BonferroniHochberg()
    pvals_array = np.array(pvals)
    expected_array = np.array(expected)
    result = bonf_hoch(pvals_array, alpha)
    assert np.array_equal(result, expected_array)

@pytest.mark.parametrize("pvals, alpha, expected", [
    ([0.01], 0.05, [True]),
    ([0.2], 0.05, [False]),
    ([0.01, 0.02], 0.05, [True, True]),
    ([0.04, 0.5], 0.05, [False, False]),
    ([0.06, 0.07, 0.2, 0.5], 0.05, [False, False, False, False]),
    ([0.0001, 0.001, 0.002, 0.003], 0.05, [True, True, True, True]),
    ([0.0005, 0.02, 0.04, 0.06, 0.2], 0.05, [True, True, False, False, False]),
    ([0.025, 0.025, 0.06, 0.06], 0.05, [True, True, False, False]),
    ([0.2, 0.15, 0.04, 0.02, 0.001], 0.05, [False, False, False, True, True]),
    ([0.05, 0.05, 0.05, 0.05], 0.05, [True, True, True, True]),
    ([0.001, 0.02, 0.025, 0.03, 0.2], 0.05, [True, True, True, True, False]),
    ([0.009, 0.5, 0.6, 0.7, 0.8], 0.05, [True, False, False, False, False])
])
def test_false_discovery_rate_correction(pvals, alpha, expected):
    fdr = FalseDiscoveryRate()
    pvals_array = np.array(pvals)
    expected_array = np.array(expected)
    result = fdr(pvals_array, alpha)
    assert np.array_equal(result, expected_array)