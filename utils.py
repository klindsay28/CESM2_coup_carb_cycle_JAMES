"""utility functions"""

import re

import cftime
import numpy as np
import xarray as xr

def clean_units(units):
    """replace some troublesome unit terms with acceptable replacements"""
    replacements = {'kgC':'kg', 'gC':'g', 'gC13':'g', 'gC14':'g', 'gN':'g',
                    'unitless':'1',
                    'years':'common_years', 'yr':'common_year',
                    'meq':'mmol', 'neq':'nmol'}
    units_split = re.split('( |\(|\)|\^|\*|/|-[0-9]+|[0-9]+)', units)
    units_split_repl = \
        [replacements[token] if token in replacements else token for token in units_split]
    return ''.join(units_split_repl)

def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        msg_full = 'unexpected dim_cnt=%d, varname=%s' % (len(ds[varname].dims), varname)
        raise ValueError(msg_full)

def time_set_mid(ds, time_name):
    """
    set ds[time_name] to midpoint of ds[time_name].attrs['bounds'], if bounds attribute exists
    type of ds[time_name] is not changed
    ds is returned
    """

    if 'bounds' not in ds[time_name].attrs:
        return ds

    # determine units and calendar of unencoded time values
    if ds[time_name].dtype == np.dtype('O'):
        units = 'days since 0000-01-01'
        calendar = 'noleap'
    else:
        units = ds[time_name].attrs['units']
        calendar = ds[time_name].attrs['calendar']

    # construct unencoded midpoint values, assumes bounds dim is 2nd
    tb_name = ds[time_name].attrs['bounds']
    if ds[tb_name].dtype == np.dtype('O'):
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
    else:
        tb_vals = ds[tb_name].values
    tb_mid = tb_vals.mean(axis=1)

    # set ds[time_name] to tb_mid
    if ds[time_name].dtype == np.dtype('O'):
        ds[time_name].values = cftime.num2date(tb_mid, units=units, calendar=calendar)
    else:
        ds[time_name].values = tb_mid

    return ds

def time_year_plus_frac(ds, time_name):
    """return time variable, as year plus fraction of year"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype('O'):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values, ds[time_name].attrs['units'], ds[time_name].attrs['calendar'])

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(tvals_cftime, 'days since 0000-01-01', calendar='noleap')

    return tvals_days / 365.0

def xr_ds_ex(encode_time=False):
    """return an example xarray.Dataset object, useful for testing functions"""

    # set up values for Dataset, 4 yrs of analytic monthly values
    days_1yr = np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    time_edges = np.insert(
        np.cumsum(np.concatenate((days_1yr, days_1yr, days_1yr, days_1yr))), 0, 0)
    time_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
    time_vals = np.mean(time_bounds_vals, axis=1)
    time_vals_yr = time_vals / 365.0
    var_vals = np.sin(np.pi * time_vals_yr) * np.exp(-0.1 * time_vals_yr)

    time_units = 'days since 0001-01-01'
    calendar = 'noleap'

    if encode_time:
        time_vals = cftime.num2date(time_vals, time_units, calendar)
        time_bounds_vals = cftime.num2date(time_bounds_vals, time_units, calendar)

    # create Dataset, including time_bounds
    time_var = xr.DataArray(time_vals, name='time', dims='time', coords={'time':time_vals},
                            attrs={'bounds':'time_bounds'})
    if not encode_time:
        time_var.attrs['units'] = time_units
        time_var.attrs['calendar'] = calendar
    time_bounds = xr.DataArray(time_bounds_vals, name='time_bounds', dims=('time', 'd2'),
                               coords={'time':time_var})
    var = xr.DataArray(var_vals, name='var', dims='time', coords={'time':time_var})
    ds = var.to_dataset()
    ds = xr.merge((ds, time_bounds))
    return ds
