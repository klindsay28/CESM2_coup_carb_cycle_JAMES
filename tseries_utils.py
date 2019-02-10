"""utility (helper) functions for handling CESM ouptut"""

import math
import os
from collections import OrderedDict

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import data_catalog

def clean_units(units):
    """replace troublesome unit terms with acceptable replacements"""
    units_ret = units
    # degC->XXXX and XXXX->degC is a kludge to avoid replacing gC with g in degC
    replacements = {'degC':'XXXX',
                    'gC':'g', 'gN':'g',
                    'year':'common_year', 'yr':'common_year',
                    'meq':'mmol', 'neq':'nmol',
                    'XXXX':'degC'}
    for key, value in replacements.items():
        units_ret = units_ret.replace(key, value)
    return units_ret

def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        msg_full = 'unexpected dim_cnt=%d, varname=%s' % (len(ds[varname].dims), varname)
        raise ValueError(msg_full)

def get_weight(ds, component, reduce_dims):
    """construct averaging/integrating weight appropriate for component and reduce_dims"""
    if component == 'lnd':
        return get_area(ds, component)
    if component == 'ice':
        return get_area(ds, component)
    if component == 'atm':
        if 'lev' in reduce_dims:
            return get_volume(ds, component)
        return get_area(ds, component)
    if component == 'ocn':
        if 'z_t' in reduce_dims or 'z_t_150m' in reduce_dims:
            return get_volume(ds, component)
        return get_area(ds, component)
    msg = 'unrecognized component=%s' % component
    raise ValueError(msg)

def get_area(ds, component):
    """return area DataArray appropriate for component"""
    if component == 'ocn':
        dim_cnt_check(ds, 'TAREA', 2)
        return ds['TAREA']
    if component == 'ice':
        dim_cnt_check(ds, 'tarea', 2)
        return ds['tarea']
    if component == 'lnd':
        dim_cnt_check(ds, 'landfrac', 2)
        dim_cnt_check(ds, 'area', 2)
        da_ret = ds['landfrac'] * ds['area']
        da_ret.name = 'area'
        da_ret.attrs['units'] = ds['area'].attrs['units']
        return da_ret
    if component == 'atm':
        dim_cnt_check(ds, 'gw', 1)
        rearth = 6.37122e6 # radius of earth used in CIME [m]
        area_earth = 4.0 * math.pi * rearth**2 # area of earth in CIME [m2]

        # normalize area so that sum over 'lat', 'lon' yields area_earth
        area = ds['gw'] + 0.0 * ds['lon'] # add 'lon' dimension
        area = (area_earth / area.sum(dim=('lat', 'lon'))) * area
        area.attrs['units'] = 'm2'
        return area
    msg = 'unknown component %s' % component
    raise ValueError(msg)

def get_volume(ds, component):
    """return volume DataArray appropriate for component"""
    msg = 'get_volume not implemented for %s' % component
    raise NotImplemented(msg)

def get_rmask(ds, component):
    """return region mask appropriate for component"""
    rmask_od = OrderedDict()
    if component == 'ocn':
        dim_cnt_check(ds, 'KMT', 2)
        lateral_dims = ds['KMT'].dims
        rmask_od['Global'] = xr.where(ds['KMT'] > 0, 1.0, 0.0)
    if component == 'ice':
        dim_cnt_check(ds, 'tmask', 2)
        dim_cnt_check(ds, 'TLAT', 2)
        lateral_dims = ds['tmask'].dims
        rmask_od['NH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'] >= 0.0), 1.0, 0.0)
        rmask_od['SH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'] < 0.0), 1.0, 0.0)
    if component == 'lnd':
        dim_cnt_check(ds, 'landfrac', 2)
        lateral_dims = ds['landfrac'].dims
        rmask_od['Global'] = xr.where(ds['landfrac'] > 0, 1.0, 0.0)
    if component == 'atm':
        dim_cnt_check(ds, 'gw', 1)
        lateral_dims = ('lat', 'lon')
        rmask_od['Global'] = xr.where((ds['lat'] > -100.0) & (ds['lon'] > -360.0), 1.0, 0.0)
    if len(rmask_od) == 0:
        msg = 'unknown component %s' % component
        raise ValueError(msg)

    rmask = xr.DataArray(np.zeros((len(rmask_od), ds.dims[lateral_dims[0]], ds.dims[lateral_dims[1]])),
                         dims=('region', lateral_dims[0], lateral_dims[1]),
                         coords={'region':list(rmask_od.keys())})

    for i, mask_logic in enumerate(rmask_od.values()):
        rmask.values[i,:,:] = mask_logic

    return rmask

def tseries_fname(varname, component, experiment, ensemble):
    """return relative filename for tseries"""
    return '%s_%s_%s_%02d.nc' % (varname, component, experiment, ensemble)

def tseries_copy_vars(component):
    """return component specific list of vars to copy into generated tseries files"""
    if component == 'atm':
        return ['co2vmr', 'ch4vmr', 'f11vmr', 'f12vmr', 'n2ovmr', 'sol_tsi']
    return []

def get_tseries(varname, component, stream, experiment):
    """
    return a tseries, as a Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    entries = data_catalog.find_in_index(
        variable=varname, component=component, stream=stream, experiment=experiment)
    paths = []
    for ensemble in entries.ensemble.unique():
        paths.append(os.path.join('tseries', tseries_fname(varname, component, experiment, ensemble)))
    if len(paths) == 1:
        ds = xr.open_dataset(paths[0], decode_times=False, decode_coords=False)
    else:
        ds = xr.open_mfdataset(paths, decode_times=False, decode_coords=False, concat_dim='ensemble')
    return ds

def time_year_plus_frac(ds, time_name):
    """return time variable, as year plus fraction of year"""
    tlen = ds.dims[time_name]
    t = np.zeros(tlen)
    for tind in range(tlen):
        date_val = ds[time_name].values[tind]
        yr = date_val.year
        yr_frac = cftime.date2num(date_val, 'days since %04d-01-01' % yr,
                                  calendar='noleap') / 365.0
        t[tind] = yr + yr_frac
    return t

def tseries_plot_1ds(ds, varnames, title, fname=None):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    t = time_year_plus_frac(ds, 'time')
    for varname in varnames:
        plt.plot(t, ds[varname], label=varname)
    plt.xlabel('time (years)')
    plt.ylabel(ds[varname].attrs['units'])
    plt.legend()
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)


def tseries_plot_1var(varname, ds_list, legend_list, title, fname=None):
    """
    create a simple plot of a tseries variables for multiple datasets
    use units from last tseries variable for ylabel
    """
    for ds_ind, ds in enumerate(ds_list):
        t = time_year_plus_frac(ds, 'time')
        plt.plot(t, ds[varname], label=legend_list[ds_ind])
    plt.xlabel('time (years)')
    plt.ylabel(ds[varname].attrs['units'])
    plt.legend()
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)
