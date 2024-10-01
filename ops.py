import torch

def gather_by_index(src, idx, dim=1, squeeze=True):
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)

def print_metric(metric):
    l = []
    for k, v in metric.items():
        if isinstance(v, int):
            r = f"{v}"
        elif isinstance(v, float):            
            if k == 'lr':
                r = f"{v:.5f}"
            else:
                r = f"{v:.2f}"
        else:
            r = v
        l.append(r)
    print(','.join(l))