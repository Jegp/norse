import torch
from norse.torch.functional.heaviside import heaviside
from norse.torch.functional.lif import LIFParameters, LIFState

from norse.torch.module.lif import (
    LIFCell,
    LIFLayer,
    LIFFeedForwardCell,
    LIFFeedForwardLayer,
)


class SNNetwork(torch.nn.Module):
    def __init__(self):
        super(SNNetwork, self).__init__()
        self.l0 = LIFCell(12, 6)
        self.l1 = LIFCell(6, 1)
        self.s0 = self.s1 = None

    def forward(self, spikes):
        spikes, self.s0 = self.l0(spikes, self.s0)
        _, self.s1 = self.l1(spikes, self.s1)
        return self.s1.v.squeeze()


def test_lif_cell():
    cell = LIFCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_cell_sparse():
    x = heaviside(torch.randn(5, 2)).to_sparse()
    p = LIFParameters(
        tau_syn_inv=torch.full((5, 4), 1.0 / 5e-3).to_sparse(),
        tau_mem_inv=torch.full((5, 4), 1.0 / 1e-2).to_sparse(),
        v_leak=torch.zeros((5, 4)).to_sparse(),
        v_th=torch.ones((5, 4)).to_sparse(),
        v_reset=torch.zeros((5, 4)).to_sparse(),
        method="super",
        alpha=torch.full((5, 4), 100).to_sparse(),
    )
    s = LIFState(
        z=torch.zeros(5, 4).to_sparse(),
        v=torch.zeros(5, 4).to_sparse(),
        i=torch.zeros(5, 4).to_sparse(),
    )
    cell = LIFCell(2, 4, p, sparsify=True)
    out, s = cell(x, s)

    for x in s:
        assert x.shape == (5, 4)
        assert x.is_sparse
    assert out.shape == (5, 4)
    assert out.is_sparse


def test_lif_layer():
    layer = LIFLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_cell_sequence():
    l1 = LIFCell(8, 6)
    l2 = LIFCell(6, 4)
    l3 = LIFCell(4, 1)
    z = torch.ones(10, 8)
    z, s1 = l1(z)
    z, s2 = l2(z)
    z, s3 = l3(z)
    assert s1.v.shape == (10, 6)
    assert s2.v.shape == (10, 4)
    assert s3.v.shape == (10, 1)
    assert z.shape == (10, 1)


def test_lif_cell_repr():
    cell = LIFCell(8, 6)
    assert (
        str(cell)
        == "LIFCell(8, 6, p=LIFParameters(tau_syn_inv=tensor(200.), tau_mem_inv=tensor(100.), v_leak=tensor(0.), v_th=tensor(1.), v_reset=tensor(0.), method='super', alpha=tensor(100.)), dt=0.001)"
    )


def test_lif_feedforward_cell():
    layer = LIFFeedForwardCell()
    data = torch.randn(5, 4)
    out, s = layer(data)

    assert out.shape == (5, 4)
    for x in s:
        assert x.shape == (5, 4)


def test_lif_feedforward_sparse():
    x = heaviside(torch.randn(5, 4)).to_sparse()
    p = LIFParameters(
        tau_syn_inv=torch.full(x.shape, 1.0 / 5e-3).to_sparse(),
        tau_mem_inv=torch.full(x.shape, 1.0 / 1e-2).to_sparse(),
        v_leak=torch.zeros(x.shape).to_sparse(),
        v_th=torch.ones(x.shape).to_sparse(),
        v_reset=torch.zeros(x.shape).to_sparse(),
        method="super",
        alpha=torch.full(x.shape, 100).to_sparse(),
    )
    layer = LIFFeedForwardCell(p)
    s = LIFState(
        z=torch.zeros(5, 4).to_sparse(),
        v=torch.zeros(5, 4).to_sparse(),
        i=torch.zeros(5, 4).to_sparse(),
    )
    out, s = layer(x, s)

    assert out.shape == (5, 4)
    assert out.is_sparse
    for x in s:
        assert x.shape == (5, 4)
        assert x.is_sparse


def test_lif_feedforward_cell_backward():
    # Tests that gradient variables can be used in subsequent applications
    cell = LIFFeedForwardCell()
    data = torch.randn(5, 4)
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()


def test_lif_feedforward_layer():
    layer = LIFFeedForwardLayer()
    data = torch.randn(10, 5, 4)
    out, s = layer(data)
    assert out.shape == (10, 5, 4)
    for x in s:
        assert x.shape == (5, 4)


def test_backward():
    model = LIFCell(12, 1)
    data = torch.ones(100, 12)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


def test_backward_iteration():
    # Tests that gradient variables can be used in subsequent applications
    model = LIFCell(6, 6)
    data = torch.ones(100, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


def test_backward_model():
    model = SNNetwork()
    data = torch.ones(100, 12)
    out = model(data)
    loss = out.sum()
    loss.backward()
