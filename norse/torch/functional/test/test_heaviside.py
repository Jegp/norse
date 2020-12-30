import torch
from norse.torch.functional.heaviside import heaviside


import torch
from pathlib import Path

torch.ops.load_library(
    Path(__file__).resolve().parent.parent.parent.parent.parent / "norse_op.so"
)


def test_heaviside():
    assert torch.equal(heaviside(torch.ones(100)), torch.ones(100))
    assert torch.equal(heaviside(-1.0 * torch.ones(100)), torch.zeros(100))


def test_heaviside_ops():
    heaviside_ops = torch.ops.norse_op.heaviside
    assert torch.equal(heaviside_ops(torch.ones(100)), torch.ones(100))
    assert torch.equal(heaviside_ops(-1.0 * torch.ones(100)), torch.zeros(100))


def test_heaviside_sparse_1d():
    i = [[0, 1, 2, 3, 4]]
    v = [0, 0.9, -1, 1, 2.9]
    s = torch.sparse_coo_tensor(i, v, (5,))
    h = heaviside(s)
    assert h.dtype == torch.uint8
    assert h.is_sparse
    assert torch.equal(h.to_dense(), torch.tensor([0, 1, 0, 1, 1]).byte())


def test_heaviside_sparse_2d():
    i = [[0, 1, 1, 1], [2, 0, 1, 2]]
    v = [0, -1, 1, 2.9]
    s = torch.sparse_coo_tensor(i, v, (2, 3))
    h = heaviside(s)
    assert h.is_sparse
    assert torch.equal(h.to_dense(), torch.tensor([[0, 0, 0], [0, 1, 1]]).byte())
