#! /usr/bin/env python3

import pytest
import cftime
import numpy as np
import numpy.matlib as npm

from src.xr_ds_ex import xr_ds_ex, gen_time_bounds_values
from src.utils import time_year_plus_frac, time_set_mid, repl_coord, da_w_lags, smooth

nyrs = 300
var_const = False


@pytest.mark.parametrize("decode_times1", [True, False])
@pytest.mark.parametrize("decode_times2", [True, False])
@pytest.mark.parametrize("apply_chunk1", [True, False])
def test_repl_coord(decode_times1, decode_times2, apply_chunk1):
    ds1 = time_set_mid(xr_ds_ex(decode_times1, nyrs=nyrs, var_const=var_const), "time")
    if apply_chunk1:
        ds1 = ds1.chunk({"time": 12})

    # change time:bounds attribute variable rename corresponding variable
    tb_name_old = ds1["time"].attrs["bounds"]
    tb_name_new = tb_name_old + "_new"
    ds1["time"].attrs["bounds"] = tb_name_new
    ds1 = ds1.rename({tb_name_old: tb_name_new})

    # verify that repl_coord on xr_ds_ex gives same results as
    # 1) executing time_set_mid
    # 2) manually changing bounds
    ds2 = repl_coord(
        "time", ds1, xr_ds_ex(decode_times2, nyrs=nyrs, var_const=var_const)
    )
    assert ds2.identical(ds1)

    assert ds2["time"].encoding == ds1["time"].encoding
    assert ds2["time"].chunks == ds1["time"].chunks


@pytest.mark.parametrize("decode_times", [True, False])
@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("apply_chunk", [True, False])
def test_time_set_mid(decode_times, deep, apply_chunk):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const, time_mid=False)
    if apply_chunk:
        ds = ds.chunk({"time": 12})

    mid_month_values = gen_time_bounds_values(nyrs).mean(axis=1)
    if decode_times:
        time_encoding = ds["time"].encoding
        expected_values = cftime.num2date(
            mid_month_values, time_encoding["units"], time_encoding["calendar"]
        )
    else:
        expected_values = mid_month_values

    ds_out = time_set_mid(ds, "time", deep)

    assert ds_out.attrs == ds.attrs
    assert ds_out.encoding == ds.encoding
    assert ds_out.chunks == ds.chunks

    for varname in ds.variables:
        assert ds_out[varname].attrs == ds[varname].attrs
        assert ds_out[varname].encoding == ds[varname].encoding
        assert ds_out[varname].chunks == ds[varname].chunks
        if varname == "time":
            assert np.all(ds_out[varname].values == expected_values)
        else:
            assert np.all(ds_out[varname].values == ds[varname].values)
            assert (ds_out[varname].data is ds[varname].data) == (not deep)

    # verify that values are independent of ds being chunked in time
    ds_chunk = xr_ds_ex(
        decode_times, nyrs=nyrs, var_const=var_const, time_mid=False
    ).chunk({"time": 6})
    ds_chunk_out = time_set_mid(ds_chunk, "time")
    assert ds_chunk_out.identical(ds_out)


@pytest.mark.parametrize("decode_times", [True, False])
def test_time_year_plus_frac(decode_times):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const)

    # call time_year_plus_frac to ensure that it doesn't raise an exception
    ty = time_year_plus_frac(ds, "time")


@pytest.mark.parametrize("decode_times", [True, False])
@pytest.mark.parametrize("add_encoding_var", [True, False])
def test_da_w_lags(decode_times, add_encoding_var):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const)

    da = ds["var_ex"]
    if add_encoding_var:
        da.encoding["_FillValue"] = 1.0e30
    lag_values = range(-12, 6 + 1, 3)
    da2 = da_w_lags(da, lag_values=lag_values)

    # verify shape, dims, attrs, and encoding of da_w_lags output
    assert da2.shape == (len(lag_values), nyrs * 12)
    assert da2.dims == ("lag",) + da.dims
    assert da2.attrs == da.attrs
    assert da2.encoding == da.encoding

    # verify proper number of fill values for each lag
    assert np.all(da2.isnull().sum("time") == abs(np.array(lag_values)))

    # verify that selecting on da_w_lags output is the same as time slice of da
    itime = 1 - min(lag_values)
    assert np.all(
        da2.isel(time=itime).values == da.isel(time=itime + np.array(lag_values))
    )


@pytest.mark.parametrize("decode_times", [True, False])
@pytest.mark.parametrize("apply_chunk", [True, False])
@pytest.mark.parametrize("add_encoding_var", [True, False])
@pytest.mark.parametrize("add_dim", [True, False])
def test_smooth(decode_times, apply_chunk, add_encoding_var, add_dim):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=True)
    if apply_chunk:
        ds = ds.chunk({"time": 12})
    da = ds["var_ex"]
    if add_dim:
        da = da.expand_dims(dim={'dim2': 2}, axis=-1)
    if add_encoding_var:
        da.encoding["_FillValue"] = 1.0e30
    da_smooth = smooth(da, 13)

    # verify shape, dims, attrs, and encoding of smooth output
    assert da_smooth.shape == da.shape
    assert da_smooth.dims == da.dims
    assert da_smooth.attrs == da.attrs
    assert da_smooth.encoding == da.encoding
    assert da_smooth.chunks == da.chunks

    # verify that non-na values are close to original values
    # this is the case because var_const=True
    assert np.all(np.isclose(da_smooth.fillna(da).values, da.values))

    # verify proper number of fill values for each lag
    print(da_smooth.load().isnull().sum("time"))
    assert np.all(da_smooth.load().isnull().sum("time").values == 12)
