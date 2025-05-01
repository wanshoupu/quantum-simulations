import pytest
import random

import numpy as np


@pytest.fixture(autouse=True)
def reset_state():
    random.seed(42)
    np.random.seed(42)
    yield
