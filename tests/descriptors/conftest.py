"""Shared parametrized tests for all registered descriptors."""

import pytest

from roxy.core.constants import AA20_ORDERED
from roxy.descriptors import DESCRIPTOR_REGISTRY

_SEQ = "".join(AA20_ORDERED)
_EMPTY = ""

_ALL_NAMES = sorted(DESCRIPTOR_REGISTRY)


@pytest.fixture(params=_ALL_NAMES)
def any_descriptor(request):
    return DESCRIPTOR_REGISTRY[request.param]()


def test_smoke(any_descriptor):
    df = any_descriptor.compute([_SEQ, _EMPTY])
    assert df.shape[0] == 2


def test_empty_schema_consistent(any_descriptor):
    assert set(any_descriptor.compute_one(_SEQ)) == set(any_descriptor.compute_one(_EMPTY))
