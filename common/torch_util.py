import torch
import numpy as np


def to_numpy(x):
    return x.detach().to('cpu').numpy()


def dict_to(x, device):
    new_x = dict()
    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        new_x[key] = value
    return new_x


def explode_shape(input, shape, dim=-1):
    """
    Reshape a 1D tensor such that it matches the shape except
    dim, index into shape
    """
    assert(isinstance(input, torch.Tensor))
    if isinstance(dim, int):
        dim = [dim]
    assert(len(input.shape) == len(dim))
    single_shape = np.ones(len(shape), dtype=np.int64)
    single_shape[dim] = input.shape
    single = input.view(*tuple(single_shape.tolist()))

    result_shape = np.array(shape, dtype=np.int64)
    result_shape[dim] = -1
    result = single.expand(*tuple(result_shape.tolist()))
    return result


def sort_select(input, *attr_tensors, 
        dim=-1, dim_slice=slice(None), 
        sort_descending=True):
    """
    Sort attributes according to input's value
    on dim. All needs to have the same number of dims
    before and at dim
    """
    last_dim = np.arange(len(input.shape))[dim] + 1
    for attr_tensor in attr_tensors:
        assert(attr_tensor.shape[:last_dim] == input.shape[:last_dim])
    selector = (slice(None),) * (last_dim - 1) + (dim_slice,)

    sorted_input, sorted_idxs = torch.sort(
        input, dim=dim, descending=sort_descending)

    result = [sorted_input[selector]]
    for attr_tensor in attr_tensors:
        selected_attr_tensor = torch.gather(
            attr_tensor, dim=dim, 
            index=sorted_idxs.expand_as(
                attr_tensor))[selector]
        result.append(selected_attr_tensor)
    return tuple(result)
