import torch
import torch_sparse


class spspmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, w):
        i = i.coalesce()
        w = w.coalesce()
        # Return immediately if one of the tensors is zero
        # I. e. if the spike input are at most zero or weights is an empty matrix
        if len(i.values()) == 0 or i.values().max() <= 0 or len(w.values()) == 0:
            return torch.sparse_coo_tensor(size=(i.shape[0], w.shape[1]))

        res_i, res_v = torch_sparse.spspmm(
            i.indices(), i.values(), w.indices(), w.values(), *i.shape, w.shape[1]
        )
        result = torch.sparse_coo_tensor(
            res_i,
            res_v,
            (i.shape[0], w.shape[1]),
            dtype=res_v.dtype,
            device=res_v.device,
        )
        ctx.save_for_backward(w)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        grad_output = grad_output.coalesce()
        wt_i, wt_v = torch_sparse.transpose(w.indices(), w.values(), *w.shape)
        out_i, out_v = torch_sparse.spspmm(
            grad_output.indices(),
            grad_output.values(),
            wt_i,
            wt_v,
            *grad_output.shape,
            w.shape[0]
        )
        r_out = torch.sparse_coo_tensor(
            out_i,
            out_v,
            (grad_output.shape[0], w.shape[0]),
            dtype=out_v.dtype,
            device=out_v.device,
        ).coalesce()
        i_out = torch.sparse_coo_tensor(
            r_out.indices(),
            torch.ones_like(r_out.values()),
            dtype=torch.uint8,
            device=out_v.device,
        )
        return r_out, i_out