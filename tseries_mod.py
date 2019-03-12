"""interface for extracting and plotting timeseries from CESM output"""

from datetime import datetime, timezone
import math
import os
import time
from collections import OrderedDict

import cf_units
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

import dask
import dask_jobqueue

import data_catalog
from utils import clean_units, dim_cnt_check, time_set_mid, time_year_plus_frac

var_specs_fname = 'var_specs.yaml'

def tseries_get(varname, component, experiment, stream=None, clobber=False):
    """
    return a tseries, as a Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode='r') as fptr:
            var_specs = yaml.load(fptr)
        stream = var_specs[component]['stream']

    # get matching data_catalog entries
    entries = data_catalog.find_in_index(
        variable=varname, component=component, stream=stream, experiment=experiment)

    # loop over matching ensembles
    paths = []
    for ensemble in entries.ensemble.unique():
        tseries_path = os.path.join('tseries', tseries_fname(varname, component, experiment, ensemble))
        tseries_path_genlock = tseries_path + ".genlock"
        # if file doesn't exists and isn't being generated, generate it
        if clobber or (not os.path.exists(tseries_path) and not os.path.exists(tseries_path_genlock)):
            # create genlock file, indicating that tseries_path is being generated
            open(tseries_path_genlock, mode='w').close()
            # generate timeseries and write it
            ds = tseries_gen(varname, component, stream, experiment, ensemble)
            ds.to_netcdf(tseries_path, unlimited_dims='time', encoding={'region':{'dtype':'S1'}})
            # remove genlock file, indicating that tseries_path has been generated
            os.remove(tseries_path_genlock)
        # wait until genlock file doesn't exists, in case it was being generated or written
        while os.path.exists(tseries_path_genlock):
            print('genlock file exists, waiting')
            time.sleep(5)
        paths.append(tseries_path)

    # if there are multiple ensembles, concatenate over ensembles
    if len(paths) > 1:
        ds = xr.open_mfdataset(paths, concat_dim='ensemble')
    else:
        ds = xr.open_dataset(paths[0])

    return ds

def tseries_plot_1ds(ds, varnames, title, region_val=None, fname=None):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    t = time_year_plus_frac(ds, 'time')
    for varname in varnames:
        if region_val is None:
            plt.plot(t, ds[varname], label=varname)
        else:
            plt.plot(t, ds[varname].sel(region=region_val), label=varname)
    plt.xlabel('time (years)')
    plt.ylabel(ds[varname].attrs['units'])
    plt.legend()
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)

def tseries_plot_1var(varname, ds_list, legend_list, title, region_val=None, fname=None):
    """
    create a simple plot of a tseries variables for multiple datasets
    use units from last tseries variable for ylabel
    """
    for ds_ind, ds in enumerate(ds_list):
        t = time_year_plus_frac(ds, 'time')
        if region_val is None:
            plt.plot(t, ds[varname], label=legend_list[ds_ind])
        else:
            plt.plot(t, ds[varname].sel(region=region_val), label=legend_list[ds_ind])
    plt.xlabel('time (years)')
    plt.ylabel(ds[varname].attrs['units'])
    plt.legend()
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)

def tseries_gen(varname, component, stream, experiment, ensemble):
    """
    generate a tseries for a particular ensemble member, return a Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    print('entering tseries_gen for %s' % varname)
    fnames = data_catalog.get_files(
        variable=varname, component=component, stream=stream, experiment=experiment, ensemble=ensemble)
    print(fnames)

    with open(var_specs_fname, mode='r') as fptr:
        var_specs_all = yaml.load(fptr)
    var_spec = var_specs_all[component]['vars'][varname]

    # use var specific reduce_dims if it exists, otherwise use reduce_dims for component
    if 'reduce_dims' in var_spec:
        reduce_dims = var_spec['reduce_dims']
    else:
        reduce_dims = var_specs_all[component]['reduce_dims']

    # get rank of varname from first file, used to set chunksize
    with xr.open_dataset(fnames[0], decode_times=False, decode_coords=False) as ds_in:
        rank = len(ds_in[varname].dims)
        tlen = ds_in.dims['time']

    time_chunksize = 12 if rank < 4 else 1
    
    # setup environment for parallel dask computations
    USER = os.environ['USER']
    processes = 4
    jobcnt = 4 if tlen <= 12*50 else 16
    project = 'P93300670'
    # create SLURM based cluster, startup workers
    cluster = dask_jobqueue.SLURMCluster(
        queue='dav', project = project, walltime = '04:00:00',
        job_extra = ['-o batch_outfiles/slurm-%j.out'],
        cores = processes, processes = processes, memory = '16GB',
        local_directory=f'/glade/scratch/{USER}/dask-tmp')
    cluster.scale(jobcnt*processes)

    # create dask distributed client, connecting to workers
    with dask.distributed.Client(cluster) as client:

        with xr.open_mfdataset(fnames, decode_times=False, decode_coords=False, data_vars='minimal',
                               chunks={'time': time_chunksize}) as ds_in:
            da_in = ds_in[varname]

            var_units = clean_units(da_in.attrs['units'])
            if 'unit_conv' in var_spec:
                var_units = '(%s)(%s)' % (str(var_spec['unit_conv']), var_units)

            # construct averaging/integrating weight
            weight = get_weight(ds_in, component, reduce_dims)
            weight_attrs = weight.attrs
            weight = get_rmask(ds_in, component) * weight
            weight.attrs = weight_attrs

            if var_spec['weight_op'] == 'integrate':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                da_out.name = varname
                da_out.attrs['long_name'] = 'Integrated '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit('(%s)(%s)' % (weight.attrs['units'], var_units)).format()
            elif var_spec['weight_op'] == 'average':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                ones_masked = xr.ones_like(da_in).where(da_in.notnull())
                denom = (ones_masked * weight).sum(dim=reduce_dims)
                da_out /= denom
                da_out.name = varname
                da_out.attrs['long_name'] = 'Averaged '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit(var_units).format()
            else:
                msg = 'weight_op==%s not implemented' % var_spec['weight_op']
                raise NotImplemented(msg)

            da_out = da_out.compute()

            if 'units_out' in var_spec:
                da_out.values = cf_units.Unit(da_out.attrs['units']).convert(
                    da_out.values, cf_units.Unit(clean_units(var_spec['units_out'])))
                da_out.attrs['units'] = var_spec['units_out']

            ds_out = da_out.to_dataset()
            merge_objs = []
            if 'bounds' in ds_in['time'].attrs:
                tb = ds_in[ds_in['time'].attrs['bounds']]
                tb.attrs['units'] = ds_in['time'].attrs['units']
                tb.attrs['calendar'] = ds_in['time'].attrs['calendar']
                ds_out = xr.merge((ds_out, tb))
            for copy_var in tseries_copy_vars(component):
                ds_out = xr.merge((ds_out, ds_in[copy_var]))
            time_set_mid(ds_out, 'time')
            ds_out.attrs = ds_in.attrs
            datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            ds_out.attrs['history'] = 'created by %s at %s' % (__file__, datestamp)

    # close cluster, as it is not needed anymore
    cluster.close()

    return ds_out.compute()

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
        KMT = ds['KMT'].fillna(0)
        rmask_od['Global'] = xr.where(KMT > 0, 1.0, 0.0)
        rmask_od['SH_mid_lat'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(100.0) < -18.0), 1.0, 0.0)
        rmask_od['low_lat'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= -18.0) & (ds['TLAT'].fillna(100.0) < 18.0), 1.0, 0.0)
        rmask_od['NH_mid_lat'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= 18.0), 1.0, 0.0)
    if component == 'ice':
        dim_cnt_check(ds, 'tmask', 2)
        dim_cnt_check(ds, 'TLAT', 2)
        lateral_dims = ds['tmask'].dims
        rmask_od['NH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'].fillna(-100.0) >= 0.0), 1.0, 0.0).compute()
        rmask_od['SH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'].fillna(100.0) < 0.0), 1.0, 0.0).compute()
    if component == 'lnd':
        dim_cnt_check(ds, 'landfrac', 2)
        lateral_dims = ds['landfrac'].dims
        rmask_od['Global'] = xr.where(ds['landfrac'] > 0, 1.0, 0.0)
        rmask_od['SH_mid_lat'] = xr.where((ds['landfrac'] > 0) & (ds['lat'] < -25.0), 1.0, 0.0)
        rmask_od['SH_low_lat'] = xr.where((ds['landfrac'] > 0) & (ds['lat'] >= -25.0) & (ds['lat'] < 0.0), 1.0, 0.0)
        rmask_od['NH_low_lat'] = xr.where((ds['landfrac'] > 0) & (ds['lat'] >= 0.0) & (ds['lat'] < 20.0), 1.0, 0.0)
        rmask_od['NH_mid_lat'] = xr.where((ds['landfrac'] > 0) & (ds['lat'] >= 20.0), 1.0, 0.0)
    if component == 'atm':
        dim_cnt_check(ds, 'gw', 1)
        lateral_dims = ('lat', 'lon')
        rmask_od['Global'] = xr.where((ds['lat'] > -100.0) & (ds['lon'] > -400.0), 1.0, 0.0)
    if len(rmask_od) == 0:
        msg = 'unknown component %s' % component
        raise ValueError(msg)

    rmask = xr.DataArray(np.zeros((len(rmask_od), ds.dims[lateral_dims[0]], ds.dims[lateral_dims[1]])),
                         dims=('region', lateral_dims[0], lateral_dims[1]),
                         coords={'region':list(rmask_od.keys())})

    # add coordinates if appropriate
    if component == 'atm' or component == 'lnd':
        rmask.coords['lat'] = ds['lat']
        rmask.coords['lon'] = ds['lon']

    for i, rmask_field in enumerate(rmask_od.values()):
        rmask.values[i,:,:] = rmask_field

    return rmask

def tseries_fname(varname, component, experiment, ensemble):
    """return relative filename for tseries"""
    return '%s_%s_%s_%02d.nc' % (varname, component, experiment, ensemble)

def tseries_copy_vars(component):
    """return component specific list of vars to copy into generated tseries files"""
    if component == 'atm':
        return ['co2vmr', 'ch4vmr', 'f11vmr', 'f12vmr', 'n2ovmr', 'sol_tsi']
    return []
