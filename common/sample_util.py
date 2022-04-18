from typing import Sequence, Tuple, Optional

import numpy as np
import zarr
import numcodecs as nc
import scipy.interpolate as si


def exp_range(start, stop, num):
    start_log = np.log(start)
    stop_log = np.log(stop)
    log_samples = np.linspace(start_log, stop_log, num)
    samples = np.exp(log_samples)
    return samples

def poly_range(start, stop, num, deg=2):
    start_p = np.power(start, 1/deg)
    end_p = np.power(stop, 1/deg)
    p_samples = np.linspace(start_p, end_p, num)
    samples = np.power(p_samples, deg)
    return samples


def get_nd_index_volume(shape):
    index_volume = np.moveaxis(np.stack(np.meshgrid(
            *(np.arange(x) for x in shape)
            , indexing='ij')), 0, -1)
    return index_volume


def get_grid_samples(*dim_samples):
    arrs = np.meshgrid(*dim_samples, indexing='ij')
    samples_volume = np.stack(arrs, axis=-1)
    return samples_volume


def get_flat_idx_samples(samples_volume, sample_dims=1):
    dim_idxs = [np.arange(x) for x in samples_volume.shape[:-sample_dims]]
    dim_idxs_volume = get_grid_samples(*dim_idxs)
    idxs_flat = dim_idxs_volume.reshape(-1, dim_idxs_volume.shape[-1])
    samples_flat = samples_volume.reshape(-1, *samples_volume.shape[-sample_dims:])
    return idxs_flat, samples_flat


class VirtualSampleGrid:
    default_data_name = 'data'

    def __init__(self, 
            axes_samples: Sequence[Tuple[str, np.array]],
            zarr_group: zarr.Group,
            compressor: Optional[nc.abc.Codec] = None):
        dim_keys = list()
        dim_samples = list()
        for key, samples in axes_samples:
            dim_keys.append(key)
            dim_samples.append(samples)
        # check uniqueness
        assert(len(set(dim_keys)) == len(dim_keys))

        if compressor is None:
            compressor = self.get_default_compressor()

        self.dim_keys = dim_keys
        self.dim_samples = dim_samples

        self.zarr_group = zarr_group
        self.compressor = compressor
    
    @classmethod
    def from_zarr_group(cls, zarr_group):
        dim_keys = zarr_group['dim_keys'][:].tolist()
        dim_samples_group = zarr_group['dim_samples']
        dim_samples = list()
        for key in dim_keys:
            dim_samples.append(dim_samples_group[key][:])
        
        compressor = None
        if cls.default_data_name in zarr_group:
            compressor = zarr_group['data'].compressor
        
        sample_grid = cls(
            zip(dim_keys, dim_samples), 
            zarr_group=zarr_group,
            compressor=compressor)
        return sample_grid
    
    @staticmethod
    def get_default_compressor():
        return nc.Blosc(
                cname='zstd', clevel=6, 
                shuffle=nc.Blosc.BITSHUFFLE)
    
    def get_sample(self, index_tuple):
        assert(len(index_tuple) == len(self.dim_samples))
        return tuple(self.dim_samples[i][j] 
            for i, j in enumerate(index_tuple))
    
    def get_idxs_volume(self, dim_ids=None):
        if dim_ids is None:
            dim_ids = list(range(len(self.shape)))
        dim_idxs = [np.arange(len(self.dim_samples[i])) 
            for i in dim_ids]
        dim_idxs_volume = get_grid_samples(*dim_idxs)
        return dim_idxs_volume
    
    def get_idxs_flat(self, dim_ids=None):
        dim_idxs_volume = self.get_idxs_volume(dim_ids=dim_ids)
        idxs_flat = dim_idxs_volume.reshape(-1, dim_idxs_volume.shape[-1])
        return idxs_flat

    def write_axes(self):
        dim_keys = self.dim_keys
        dim_samples = self.dim_samples
        zarr_group = self.zarr_group
        compressor = self.compressor

        dim_samples_group = zarr_group.require_group(
            'dim_samples', overwrite=False)
        for key, value in zip(dim_keys, dim_samples):
            dim_samples_group.array(
                name=key, data=value, chunks=value.shape, 
                compressor=compressor, overwrite=True)
        zarr_group.array(name='dim_keys', data=dim_keys, 
            compressor=compressor, overwrite=True)
    
    def allocate_data(self, 
            name=None, 
            grid_shape=None, data_shape=tuple(), dtype=np.float32, 
            fill_value=np.nan, overwrite=False, compressor=None):
        zarr_group = self.zarr_group
        if compressor is None:
            compressor = self.compressor
        if name is None:
            name = self.default_data_name
        if grid_shape is None:
            grid_shape = self.shape

        data_array = zarr_group.require_dataset(
            name=name, dtype=dtype, compressor=compressor,
            shape=grid_shape + data_shape, 
            chunks=(1,) * len(grid_shape) + data_shape,
            fill_value=fill_value,
            overwrite=overwrite)
        return data_array
    
    @property
    def shape(self):
        return tuple(len(x) for x in self.dim_samples)


class NdGridInterpolator:
    def __init__(self, 
            grid: np.array, 
            dim_samples: Sequence[np.array],
            seed: int = 0):
        assert(grid.shape == tuple(len(x) for x in dim_samples))
        grid_coords = np.vstack(np.nonzero(grid)).T
        dim_interpolators = [si.interp1d(
            x=np.arange(len(y)),
            y=y, fill_value='extrapolate')
            for y in dim_samples]
        rs = np.random.RandomState(seed=seed)

        self.grid_coords = grid_coords
        self.dim_interpolators = dim_interpolators
        self.rs = rs
    
    def get_coord_sample(self, size: int) -> np.ndarray:
        grid_coords = self.grid_coords
        rs = self.rs
        coord_idx_sample = rs.choice(len(grid_coords), size=size)
        diffuse_sample = rs.uniform(0, 1, 
            size=(size, grid_coords.shape[-1]))
        coord_sample = grid_coords[coord_idx_sample] + diffuse_sample
        return coord_sample
        
    def get_sample(self, size: int) -> np.ndarray:
        dim_interpolators = self.dim_interpolators
        
        coord_sample = self.get_coord_sample(size)
        sample = np.zeros_like(coord_sample)
        for i in range(coord_sample.shape[-1]):
            sample[:,i] = dim_interpolators[i](coord_sample[:,i])
        return sample


class GridCoordTransformer:
    def __init__(self, 
        low: tuple, high:tuple,
        grid_shape: tuple):
        self.kwargs = {
            'low': low,
            'high': high,
            'grid_shape': grid_shape
        }

        low = np.array(low)
        high = np.array(high)
        grid_shape = np.array(grid_shape)

        # pixel per meter
        scale = grid_shape / (high - low)

        # offset + real = pixel origin
        offset = - low
    
        self.scale = scale
        self.offset = offset
        self.grid_shape = grid_shape
        
    def to_zarr(self, zarr_path):
        root = zarr.open(zarr_path, 'rw')
        attrs = root.attrs.asdict()
        attrs['transformer'] = self.kwargs
        root.attrs.put(attrs)

    @classmethod
    def from_zarr(cls, zarr_path):
        root = zarr.open(zarr_path, 'r')
        attrs = root.attrs.asdict()
        kwargs = attrs['transformer']
        return cls(**kwargs)
    
    @property
    def shape(self):
        return tuple(self.grid_shape.tolist())
    
    @property
    def pix_per_m(self):
        return np.mean(self.scale)

    def to_grid(self, coords, clip=True):
        offset = self.offset
        scale = self.scale
        grid_shape = self.grid_shape

        result = (coords + offset) * scale
        if clip:
            result = np.clip(
                result, a_min=(0,0), a_max=grid_shape)
        return result
    
    def from_grid(self, coords):
        offset = self.offset
        scale = self.scale

        result = (coords / scale) - offset
        return result


def ceil_div(a, b):
    return -(-a // b)

class ArraySlicer:
    def __init__(self, shape: tuple, chunks: tuple):
        assert(len(chunks) <= len(shape))
        relevent_shape = shape[:len(chunks)]
        chunk_size = tuple(ceil_div(*x) \
            for x in zip(relevent_shape, chunks))
        
        self.relevent_shape = relevent_shape
        self.chunks = chunks
        self.chunk_size = chunk_size
    
    def __len__(self):
        chunk_size = self.chunk_size
        return int(np.prod(chunk_size))
    
    def __getitem__(self, idx):
        relevent_shape = self.relevent_shape
        chunks = self.chunks
        chunk_size = self.chunk_size
        chunk_stride = np.cumprod((chunk_size[1:] + (1,))[::-1])[::-1]
        chunk_idx = list()
        mod = idx
        for x in chunk_stride:
            chunk_idx.append(mod // x)
            mod = mod % x

        slices = list()
        for i in range(len(chunk_idx)):
            start = chunks[i] * chunk_idx[i]
            end = min(relevent_shape[i], 
                chunks[i] * (chunk_idx[i] + 1))
            slices.append(slice(start, end))
        return slices

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def transpose_data_dict(data):
    assert(len(data) > 0)
    result = dict()
    for key in data[0].keys():
        if isinstance(data[0][key], list):
            result[key] = [x[key] for x in data]
        elif isinstance(data[0][key], np.ndarray):
            same_shape = True
            shape = data[0][key].shape
            for x in data:
                if x[key].shape != shape:
                    same_shape = False
            if same_shape:
                result[key] = np.array([x[key] for x in data])
            else:
                result[key] = [x[key] for x in data]
        else:
            result[key] = np.array([x[key] for x in data])
    return result
