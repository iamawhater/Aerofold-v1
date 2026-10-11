import pytest
import torch

from backend.demo import make_demo


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def demo():
    return make_demo()[0]
