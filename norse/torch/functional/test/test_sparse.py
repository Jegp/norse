import torch
import pytest

from norse.torch.functional.sparse import spspmm

def test_spspmm():
    m1 = torch.randn(2, 2)
    m2 = torch.randn(2, 3)
    assert torch.equal(m1 @ m2, spspmm.apply(m1.to_sparse(), m2.to_sparse()).to_dense())

def test_spspmm_empty():
    m1 = torch.zeros(2, 3)
    m2 = torch.randn(3, 5)
    assert torch.equal(m1 @ m2, spspmm.apply(m1.to_sparse(), m2.to_sparse()).to_dense())

def test_spspmm_empty_with_indices():
    m1 = torch.sparse_coo_tensor(torch.tensor([[0], [1]], dtype=torch.int32), torch.tensor([0.0]), (2, 3))
    m2 = torch.randn(3, 5)
    assert torch.equal(m1 @ m2, spspmm.apply(m1, m2.to_sparse()).to_dense())

def test_spspmm_empty_weights():
    m1 = torch.randn(2, 3)
    m2 = torch.zeros(3, 5)
    assert torch.equal(m1 @ m2, spspmm.apply(m1.to_sparse(), m2.to_sparse()).to_dense())

def test_spspmm_back():
    i_dense = torch.tensor([[0, 1.0]])
    w = torch.tensor([[1.5, 5], [3, 2], [3, -1]])
    i_dense.requires_grad = True
    out_dense = (i_dense @ w.t())
    out_dense.sum().backward()

    i_sparse = torch.tensor([[0, 1.0]]).to_sparse()
    i_sparse.requires_grad = True
    out_sparse = spspmm.apply(i_sparse, w.t().to_sparse())
    out_sparse.to_dense().sum().backward()
    
    assert torch.equal(out_dense, out_sparse.to_dense())
    assert torch.equal(i_dense.grad, i_sparse.grad.to_dense())