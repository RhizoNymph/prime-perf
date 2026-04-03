import pytest

from perf_optimize.reward import PERF_WEIGHT_MAP


def test_perf_weight_map_is_immutable():
    with pytest.raises(TypeError):
        PERF_WEIGHT_MAP["cycles"] = 999
