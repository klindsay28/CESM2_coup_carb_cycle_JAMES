"""function for example xarray.Dataset object"""

import cftime
import numpy as np
import numpy.matlib as npm
import xarray as xr

def xr_ds_ex(decode_times=True, nyrs=3, var_const=True):
    """return an example xarray.Dataset object, useful for testing functions"""

    # set up values for Dataset, 4 yrs of analytic monthly values
    days_1yr = np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    time_edges = np.insert(
        np.cumsum(npm.repmat(days_1yr, nyrs, 1)), 0, 0)
    time_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
    time_vals = 0.25 * time_bounds_vals[:,0] + 0.75 * time_bounds_vals[:,1]
    time_vals_yr = time_vals / 365.0
    if var_const:
        var_vals = np.ones_like(time_vals_yr)
    else:
        var_vals = np.sin(np.pi * time_vals_yr) * np.exp(-0.1 * time_vals_yr)

    time_units = 'days since 0001-01-01'
    calendar = 'noleap'

    if decode_times:
        time_vals = cftime.num2date(time_vals, time_units, calendar)
        time_bounds_vals = cftime.num2date(time_bounds_vals, time_units, calendar)

    # create Dataset, including time_bounds
    time_var = xr.DataArray(time_vals, name='time', dims='time', coords={'time':time_vals},
                            attrs={'bounds':'time_bounds'})
    if not decode_times:
        time_var.attrs['units'] = time_units
        time_var.attrs['calendar'] = calendar
    time_bounds = xr.DataArray(time_bounds_vals, name='time_bounds', dims=('time', 'd2'),
                               coords={'time':time_var})
    var = xr.DataArray(var_vals, name='var_ex', dims='time', coords={'time':time_var})
    ds = var.to_dataset()
    ds = xr.merge([ds, time_bounds])

    if decode_times:
        ds.time.encoding['units'] = time_units
        ds.time.encoding['calendar'] = calendar

    return ds
