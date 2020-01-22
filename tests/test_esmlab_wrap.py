#! /usr/bin/env python3

import time

import pytest
import cftime
import numpy as np
import numpy.matlib as npm

from src.esmlab_wrap import compute_ann_mean
from src.utils import time_year_plus_frac
from src.xr_ds_ex import xr_ds_ex
from src.utils_test import dict_skip_keys, ds_identical_skip_attr_list

nyrs = 3

time_edges = np.insert(np.cumsum(npm.repmat([365.0], nyrs, 1)), 0, 0)
year_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
year_mid_vals = year_bounds_vals.mean(axis=1)

@pytest.mark.parametrize('decode_times', [True, False])
@pytest.mark.parametrize('add_encoding_var', [True, False])
@pytest.mark.parametrize('unlimited_dim', [True, False])
@pytest.mark.parametrize('var_const', [True, False])
def test_compute_ann_mean(decode_times, add_encoding_var, unlimited_dim, var_const):
    print(f'decode_times={decode_times}')
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const)
    if add_encoding_var:
        ds['var_ex'].encoding['_FillValue'] = 1.0e30
    if unlimited_dim:
        ds.encoding['unlimited_dims'] = 'time'
    if not var_const:
        # values whose weighted average is identically 0.0
        ds['var_ex'].values[0:12] = \
            np.array([ 30.0,   0.0,  30.0,  -31.0,  30.0,  -31.0,
                       30.0,   0.0,  -31.0,   0.0,  -31.0,   0.0])

    ds_out = compute_ann_mean(ds)

    # verify dims, attrs, and encoding are preserved for all variables
    for varname in ds.variables:
        assert ds_out[varname].dims == ds_out[varname].dims
        assert ds_out[varname].attrs == ds[varname].attrs
        assert ds_out[varname].encoding == ds[varname].encoding

    # verify global (non-history) attrs and encoding are preserved
    skip_attr_list = ['history']
    assert dict_skip_keys(ds_out.attrs, skip_attr_list) \
        == dict_skip_keys(ds.attrs, skip_attr_list)
    assert ds_out.encoding == ds.encoding

    # verify compute_ann_mean time:bounds and time values
    if decode_times:
        units = ds_out['time'].encoding['units']
        calendar = ds_out['time'].encoding['calendar']
        target_year_bounds_vals = \
            cftime.num2date(year_bounds_vals, units, calendar)
        target_time_vals = cftime.num2date(year_mid_vals, units, calendar)
    else:
        target_year_bounds_vals = year_bounds_vals
        target_time_vals = year_mid_vals
    tb_name = ds['time'].attrs['bounds']
    assert np.all(ds_out[tb_name].values == target_year_bounds_vals)
    assert np.all(ds_out['time'].values == target_time_vals)

    # verify compute_ann_mean var_ex values are correct
    if var_const:
        assert np.all(ds_out['var_ex'].values == 1.0)
    else:
        assert np.all(ds_out['var_ex'].values[0] == 0.0)

    # verify that results are the same if ds is chunked in time
    # sleep, to ensure a different timestamp in history attribute
    time.sleep(1)
    ds_chunk_out = compute_ann_mean(ds.chunk({'time': 12}))
    assert ds_chunk_out.attrs['history'] != ds_out.attrs['history']
    skip_attr_list = ['history']
    assert ds_identical_skip_attr_list(ds_out, ds_chunk_out, skip_attr_list)
