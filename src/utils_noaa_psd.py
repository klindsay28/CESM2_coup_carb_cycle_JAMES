"""utilities for NOAA PSD format"""

import cftime
import numpy as np
import xarray as xr

from src.xr_ds_ex import gen_time_bounds_values


def psd_read_file(
    fname_psd, name, attrs, encode_time=False, calendar="noleap", year_ref=1
):
    """
    Return a Dataset of values from a NOAA PSD formatted file.
    If encode_time==False, time values are cftime objects.
    Otherwise they are floating point numbers in units of "days since {year_ref}-01-01",
    with calendar=calendar.
    """
    with open(fname_psd) as fobj:
        # get year range from 1st line
        vals_as_strs = fobj.readline().split()
        year_beg = int(vals_as_strs[0])
        year_end = int(vals_as_strs[1])
        ds = gen_monthly_time_vars(year_beg, year_end, encode_time, calendar, year_ref)

        # read values, line-by-line
        year_cnt = year_end - year_beg + 1
        var_values = np.empty((year_cnt, 12))
        for year in range(year_beg, year_end + 1):
            vals_as_strs = fobj.readline().split()
            var_values[year - year_beg, :] = np.array(vals_as_strs[1:])
        var_values = var_values.reshape(12 * year_cnt)

    # create DataArray and add to Dataset
    da = xr.DataArray(var_values, dims="time", attrs=attrs)
    ds[name] = da

    ds.attrs["input_file_list"] = fname_psd

    return ds


def gen_monthly_time_vars(
    year_beg, year_end, encode_time=False, calendar="noleap", year_ref=1
):
    """
    Return Dataset with monthly time and time_bounds DataArrays spanning year_beg to year_end
    Generated values include year_end.
    If encode_time==False, time values are cftime objects.
    Otherwise they are floating point numbers in units of "days since {year_ref}-01-01",
    with calendar=calendar.
    """

    # select which cftime class will be used, based on calendar
    cftime_class = None
    if calendar in ["gregorian", "standard"]:
        cftime_class = cftime.DatetimeGregorian
    if calendar in ["proleptic_gregorian"]:
        cftime_class = cftime.DatetimeProlepticGregorian
    if calendar in ["noleap", "365_day"]:
        cftime_class = cftime.DatetimeNoLeap
    if cftime_class is None:
        raise NotImplementedError(f"calendar={calendar} not implemented")

    # construct decoded, i.e. arrays of cftime objects, time values
    year_cnt = year_end - year_beg + 1
    time_edges_values = np.empty(12 * year_cnt + 1, dtype=np.dtype("O"))
    ind = 0
    for year in range(year_beg, year_end + 1):
        for mon in range(1, 13):
            time_edges_values[ind] = cftime_class(year, mon, 1)
            ind += 1
    time_edges_values[ind] = cftime_class(year_end + 1, 1, 1)

    time_bounds_values = np.stack(
        (time_edges_values[:-1], time_edges_values[1:]), axis=1
    )
    time_values = time_bounds_values[:, 0] + 0.5 * (
        time_bounds_values[:, 1] - time_bounds_values[:, 0]
    )

    time_units = f"days since {year_ref:04d}-01-01"

    # encode time, if requested
    if encode_time == True:
        time_bounds_values = cftime.date2num(
            time_bounds_values, units=time_units, calendar=calendar
        )
        time_values = cftime.date2num(time_values, units=time_units, calendar=calendar)

    time_var = xr.DataArray(
        time_values,
        name="time",
        dims="time",
        coords={"time": time_values},
        attrs={"bounds": "time_bounds"},
    )
    time_bounds_var = xr.DataArray(
        time_bounds_values,
        name="time_bounds",
        dims=("time", "d2"),
        coords={"time": time_var},
    )

    # store metadata appropriately
    if encode_time == True:
        time_var.attrs["units"] = time_units
        time_var.attrs["calendar"] = calendar
    else:
        time_var.encoding["units"] = time_units
        time_var.encoding["calendar"] = calendar

    ds = xr.Dataset({"time": time_var, "time_bounds": time_bounds_var})
    return ds
