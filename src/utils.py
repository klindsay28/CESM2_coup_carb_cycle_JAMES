"""utility functions"""

from datetime import datetime
import inspect

import dask

import cftime
import cf_units
import numpy as np
import xarray as xr


def print_timestamp(msg):
    print(f"{str(datetime.now())}({inspect.stack()[1][3]}):{msg}")


def is_date(da):
    """
    Determine if da is a date-like variable.
    If da values are objects, only datetime objects are recognized as date-like.
    Otherwise, the units attribute is checked to see if it is date-like,
        using cf-units's is_time_reference.
    Return False if da values are not objects and da has no units attribute.
    """
    if da.dtype == np.dtype("O"):
        return isinstance(da.values[0], cftime.datetime)
    if "units" in da.attrs:
        return cf_units.Unit(da.attrs["units"]).is_time_reference()
    return False


def repl_coord(coordname, ds1, ds2):
    """
    Return copy of d2 with coordinate coordname replaced, using coordname from ds1.
    Drop ds2.coordname.attrs['bounds'] in result, if ds2.coordname has bounds attribute.
    Add ds1.coordname.attrs['bounds'] to result, if ds1.coordname has bounds attribute.
    Except for coordname, the returned Dataset is a non-deep copy of ds2.
    """
    if "bounds" in ds2[coordname].attrs:
        tb_name = ds2[coordname].attrs["bounds"]
        ds_out = ds2.drop(tb_name).assign_coords({coordname: ds1[coordname]})
    else:
        ds_out = ds2.assign_coords({coordname: ds1[coordname]})
    if "bounds" in ds1[coordname].attrs:
        tb_name = ds1[coordname].attrs["bounds"]
        ds_out = xr.merge([ds_out, ds1[tb_name]])
    return ds_out


def copy_fill_settings(da_in, da_out):
    """
    propagate _FillValue and missing_value settings from da_in to da_out
    return da_out
    """
    for key in ["_FillValue", "missing_value"]:
        if key in da_in.encoding:
            da_out.encoding[key] = da_in.encoding[key]
    return da_out


def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        msg_full = "unexpected dim_cnt=%d, varname=%s" % (
            len(ds[varname].dims),
            varname,
        )
        raise ValueError(msg_full)


def time_set_mid(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with midpoints of
    ds[time_name].attrs['bounds'], if bounds attribute exists.
    Except for time_name, the returned Dataset is a copy of ds2.
    The copy is deep or not depending on the argument deep.
    """

    ds_out = ds.copy(deep)

    if "bounds" not in ds[time_name].attrs:
        return ds_out

    tb_name = ds[time_name].attrs["bounds"]
    tb = ds[tb_name]
    bounds_dim = next(dim for dim in tb.dims if dim != time_name)

    # Use da = da.copy(data=...), in order to preserve attributes and encoding.

    # If tb is an array of datetime objects then encode time before averaging.
    # Do this because computing the mean on datetime objects with xarray fails
    # if the time span is 293 or more years.
    #     https://github.com/klindsay28/CESM2_coup_carb_cycle_JAMES/issues/7
    if tb.dtype == np.dtype("O"):
        units = "days since 0001-01-01"
        calendar = "noleap"
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
        tb_mid_decode = cftime.num2date(
            tb_vals.mean(axis=1), units=units, calendar=calendar
        )
        ds_out[time_name] = ds[time_name].copy(data=tb_mid_decode)
    else:
        ds_out[time_name] = ds[time_name].copy(data=tb.mean(bounds_dim))

    return ds_out


def time_year_plus_frac(ds, time_name):
    """return time variable, as numpy array of year plus fraction of year values"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype("O"):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values,
            ds[time_name].attrs["units"],
            ds[time_name].attrs["calendar"],
        )

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(
        tvals_cftime, "days since 0000-01-01", calendar="noleap"
    )

    return tvals_days / 365.0


def smooth_1d_np(vals, filter_len=10 * 12, ret_edge_len=False):
    if filter_len % 2 == 1:
        smooth_edge_len = (filter_len - 1) // 2
        w = np.ones(filter_len)
    else:
        smooth_edge_len = filter_len // 2
        w = np.ones(filter_len + 1)
        w[0] = 0.5
        w[-1] = 0.5
    w *= 1.0 / sum(w)
    ret_val = np.convolve(w, vals, mode="same")
    ret_val[0:smooth_edge_len] = np.nan
    ret_val[-1 : -1 - smooth_edge_len : -1] = np.nan
    if ret_edge_len:
        return ret_val, smooth_edge_len
    else:
        return ret_val


def smooth(da, filter_len=10 * 12, ret_edge_len=False):
    """apply smooth_1d_np to da values along leading dimension of da"""
    da_out = da.copy()
    if len(da_out.dims) == 1:
        da_out.values, smooth_edge_len = smooth_1d_np(
            da_out.values, filter_len, ret_edge_len=True
        )
    else:
        da_stack = da_out.stack(stackdim=da_out.dims[1:])
        for stackdim_ind in range(da_stack.shape[-1]):
            da_stack.values[:, stackdim_ind], smooth_edge_len = smooth_1d_np(
                da_stack.values[:, stackdim_ind], filter_len, ret_edge_len=True
            )
    if ret_edge_len:
        return da_out, smooth_edge_len
    else:
        return da_out


def da_normalize(da):
    """normalize da values along leading dimension"""
    dimname = da.dims[0]
    da_out = da.copy()
    da_out -= da.mean(dimname)
    da_out /= da_out.std(dimname)
    da_out.attrs["long_name"] = " ".join(
        [da_out.attrs["long_name"], f"(normalized along {dimname} dimension)"]
    )
    da_out.attrs["units"] = "1"
    return da_out


def da_w_lags(da, lag_values=range(-36, 37)):
    """
    return da with added lag dimension
    lagging is done along da's leading dimension
    positive lag shifts to the left
        e.g., if dimension is time, positive lag samples the future
    """
    dimname = da.dims[0]
    da_out = da.expand_dims(dim={"lag": lag_values}).copy()
    for lag_ind, lag_val in enumerate(lag_values):
        da_out.values[lag_ind, :] = da.shift({dimname: -lag_val}).values
    return da_out


def copy_var_names(component):
    """return component specific list of vars to copy into generated Datasets"""
    if component == "atm":
        return [
            "P0",
            "hyai",
            "hyam",
            "hybi",
            "hybm",
            "co2vmr",
            "ch4vmr",
            "f11vmr",
            "f12vmr",
            "n2ovmr",
            "sol_tsi",
        ]
    return []


def drop_var_names(component, ds, varname):
    """return component/Dataset specific list of vars to drop when opening netcdf files"""
    if component == "lnd":
        retval = []
        Time_constant_3Dvars = "ZSOI:DZSOI:WATSAT:SUCSAT:BSW:HKSAT:ZLAKE:DZLAKE"
        for varname in Time_constant_3Dvars.split(":"):
            if varname in ds:
                retval.append(varname)
        return retval
    if component == "ocn":
        # drop non-used coordinates and variables that use them
        # their presence causes xarray to generate incorrect coordinate attributes
        coordnames = ["TLAT", "TLONG", "ULAT", "ULONG"]
        drop_coordnames = [
            coordname
            for coordname in coordnames
            if coordname not in ds[varname].encoding["coordinates"]
        ]
        retval = drop_coordnames
        vars_on_drop_coordnames = [
            varname_tmp
            for varname_tmp in ds.variables
            if "coordinates" in ds[varname_tmp].encoding
            and (
                set(drop_coordnames)
                & set(ds[varname_tmp].encoding["coordinates"].split(" "))
            )
        ]
        retval.extend(vars_on_drop_coordnames)
        return retval
    return []
