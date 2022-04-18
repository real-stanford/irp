import os

import zarr
import numpy as np

from common.sample_util import get_nd_index_volume


def get_hybrid_coordinate_selection(
        array: zarr.Array, 
        indices: np.ndarray, 
        index_dims: tuple):
    """
    Example
    array: (50,50,50,10,256,256)
    indices: (256,256,10,5)
    index_dims: (0,1,2,-2,-1)
    return: (256,256,10,10)
    """
    assert(indices.shape[-1] == len(index_dims))
    flat_indices = indices.reshape(-1, indices.shape[-1])

    array_shape = np.array(array.shape)
    dim_is_index = np.zeros(len(array.shape), dtype=bool)
    dim_is_index[index_dims] = True

    item_shape = array_shape[~dim_is_index]
    item_size = np.prod(item_shape)
    
    item_coords = get_nd_index_volume(item_shape)
    num_coords = len(flat_indices) * item_size

    coords = np.zeros(
        (len(array.shape), num_coords), 
        dtype=np.uint64)
    coords[dim_is_index] = np.repeat(
        flat_indices, item_size, axis=0).T
    coords[~dim_is_index] = np.tile(
        item_coords, (flat_indices.shape[0],1)).T
    
    # get data
    data = array.get_coordinate_selection(tuple(coords))

    # reshape result
    result = data.reshape(
        tuple(indices.shape[:-1]) + tuple(item_shape))
    
    return result


def get_initialized_chunk_coords(
        array: zarr.Array, sort=True):
    n_dims = len(array.shape)
    array_dir = os.path.join(array.store.path, array.path)
    fnames = os.listdir(array_dir)

    coords_list = list()
    for fname in fnames:
        this_strs = fname.split('.')
        if len(this_strs) == n_dims:
            coords_list.append([int(x) for x in this_strs])
    coords = np.array(coords_list, dtype=np.int64)
    if sort:
        sorted_idxs = np.lexsort(coords.T[::-1], axis=-1)
        coords = coords[sorted_idxs]
    return coords


def get_is_initialized_volume(
        array: zarr.Array, dims=None):
    if dims is None:
        dims = np.arange(len(array.shape))[
            np.array(array.chunks) == 1]
    
    coords = get_initialized_chunk_coords(
        array=array, sort=False)
    selected_shape = tuple(np.array(array.shape)[dims])
    if len(coords) == 0:
        return np.zeros(selected_shape, dtype=bool)

    selected_coords = coords[:,dims]
    is_initialized_volume = np.zeros(selected_shape, dtype=bool)
    is_initialized_volume[tuple(selected_coords.T)] = True
    return is_initialized_volume


def parse_bytes(s):
    """Parse byte string to numbers

    >>> from dask.utils import parse_bytes
    >>> parse_bytes('100')
    100
    >>> parse_bytes('100 MB')
    100000000
    >>> parse_bytes('100M')
    100000000
    >>> parse_bytes('5kB')
    5000
    >>> parse_bytes('5.4 kB')
    5400
    >>> parse_bytes('1kiB')
    1024
    >>> parse_bytes('1e6')
    1000000
    >>> parse_bytes('1e6 kB')
    1000000000
    >>> parse_bytes('MB')
    1000000
    >>> parse_bytes(123)
    123
    >>> parse_bytes('5 foos')
    Traceback (most recent call last):
        ...
    ValueError: Could not interpret 'foos' as a byte unit
    """
    if isinstance(s, (int, float)):
        return int(s)
    s = s.replace(" ", "")
    if not any(char.isdigit() for char in s):
        s = "1" + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    try:
        n = float(prefix)
    except ValueError as e:
        raise ValueError("Could not interpret '%s' as a number" % prefix) from e

    try:
        multiplier = byte_sizes[suffix.lower()]
    except KeyError as e:
        raise ValueError("Could not interpret '%s' as a byte unit" % suffix) from e

    result = n * multiplier
    return int(result)


byte_sizes = {
    "kB": 10 ** 3,
    "MB": 10 ** 6,
    "GB": 10 ** 9,
    "TB": 10 ** 12,
    "PB": 10 ** 15,
    "KiB": 2 ** 10,
    "MiB": 2 ** 20,
    "GiB": 2 ** 30,
    "TiB": 2 ** 40,
    "PiB": 2 ** 50,
    "B": 1,
    "": 1,
}
byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and "i" not in k})
byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and "i" in k})


def open_cached(zarr_path, mode='a', cache_size='4GB', **kwargs):
    cache_bytes = 0
    if cache_size:
        cache_bytes = parse_bytes(cache_size)
    
    store = zarr.DirectoryStore(zarr_path)
    chunk_store = store
    if cache_bytes > 0:
        chunk_store = zarr.LRUStoreCache(store, max_size=cache_bytes)
    
    group = zarr.open_group(store=store, mode=mode, chunk_store=chunk_store, **kwargs)
    return group


def require_parent_group(group, name, overwrite=False):
    parent_name = os.path.dirname(name)
    if parent_name == '':
        return group
    if parent_name in group:
        pg = group[parent_name]
        if isinstance(pg,zarr.hierarchy.Group):
            return pg
        raise RuntimeError('Group {} already exists!'.format(parent_name))
    pg = require_parent_group(group, parent_name, overwrite=overwrite)
    result = pg.require_group(os.path.basename(name), overwrite=overwrite)
    return result
