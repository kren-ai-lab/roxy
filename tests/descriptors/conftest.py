"""Shared parametrized tests for all registered descriptors."""

import pytest

from roxy.descriptors import DESCRIPTOR_REGISTRY

from ._helpers import EMPTY, SEQ_ALL20

_ALL_NAMES = sorted(DESCRIPTOR_REGISTRY)


@pytest.fixture(params=_ALL_NAMES)
def any_descriptor(request):
    return DESCRIPTOR_REGISTRY[request.param]()


def test_smoke(any_descriptor):
    df = any_descriptor.compute([SEQ_ALL20, EMPTY])
    assert df.shape[0] == 2


def test_empty_schema_consistent(any_descriptor):
    assert set(any_descriptor.compute_one(SEQ_ALL20)) == set(any_descriptor.compute_one(EMPTY))


def test_compute_prefixes_features_with_registry_name(any_descriptor):
    features = any_descriptor.compute_one(SEQ_ALL20)
    df = any_descriptor.compute([SEQ_ALL20], ids=["s1"])

    assert df.columns == ["id", *[f"{any_descriptor.name}_{key}" for key in features]]
