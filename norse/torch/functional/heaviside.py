import torch


@torch.jit.script
def heaviside(data):
    """
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.

    .. math::
        H[n]=\\begin{cases} 0, & n <= 0, \\ 1, & n \\g 0, \\end{cases}
    """
    if data.is_sparse:
        coalesced = data.coalesce()
        values = coalesced.values()
        indices = coalesced.indices()
        filter_gt = torch.gt(values, torch.as_tensor(0.0))
        v = torch.ones_like(values, dtype=torch.uint8)[filter_gt]
        ndims = len(data.shape)
        i = indices[filter_gt.view(1, -1).expand(ndims, -1)].reshape(ndims, -1)
        return torch.sparse_coo_tensor(i, v, data.shape, device=data.device)

    return torch.gt(data, torch.as_tensor(0.0)).to(data.dtype)
