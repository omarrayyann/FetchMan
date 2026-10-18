

def make_broadcastable(source, target_shape, prepend: bool = False):
    """Adds singleton dimensions to the tensor to make it broadcastable to the given shape."""
    dims_to_add = len(target_shape) - source.ndim
    if prepend:
        assert source.shape == target_shape[-source.ndim:], 'Source tensor must match last dimensions of target shape!'
        return source.view(*([1] * dims_to_add), *source.shape)
    return source.view(*source.shape, *([1] * dims_to_add))
