import json

import pytest

from quompiler.construct.types import QompilePlatform
from quompiler.config.construct import QompilePlatformEncoder, qompile_platform_decoder


@pytest.mark.parametrize("name, expected", [
    ['Cirq', QompilePlatform.CIRQ],
])
def test_QompilePlatform_serde(name, expected):
    # Serialize
    json_str = json.dumps({"platform": QompilePlatform.CIRQ}, cls=QompilePlatformEncoder)

    # Deserialize
    platform_obj = json.loads(json_str, object_hook=qompile_platform_decoder)

    # verify
    actual = platform_obj["platform"]
    assert actual == expected
