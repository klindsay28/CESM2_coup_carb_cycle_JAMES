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
# import dask_jobqueue
import ncar_jobqueue

import data_catalog
from utils import clean_units, copy_fill_settings, dim_cnt_check, time_set_mid, time_year_plus_frac

var_specs_fname = 'var_specs.yaml'
time_name = 'time'

def tseries_get_vars(varnames, component, experiment, stream=None, clobber=None, cluster_in=None):
    """
    return tseries for varnames, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called

    arguments are passed to tseries_get_var
    """
    if clobber is None:
        clobber = os.environ['CLOBBER'] == 'True' if 'CLOBBER' in os.environ else False
    cluster = ncar_jobqueue.NCARCluster() if cluster_in is None and clobber else cluster_in
    for varind, varname in enumerate(varnames):
        ds_tmp = tseries_get_var(varname, component, experiment, stream, clobber, cluster)
        if varind == 0:
            ds = ds_tmp
        else:
            ds[varname] = ds_tmp[varname]
    if cluster_in is None and clobber:
        cluster.close()
    return ds

def tseries_get_var(varname, component, experiment, stream=None, clobber=None, cluster_in=None):
    """
    return tseries for varname, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode='r') as fptr:
            var_specs = yaml.safe_load(fptr)
        stream = var_specs[component]['stream']

    # get matching data_catalog entries
    entries = data_catalog.find_in_index(
        variable=_varname_resolved(varname, component), component=component,
        stream=stream, experiment=experiment)

    # if clobber is not specified via argument, check for specification via environment
    # if not specified via environment, default value is False
    if clobber is None:
        clobber = os.environ['CLOBBER'] == 'True' if 'CLOBBER' in os.environ else False

    # loop over matching ensembles
    paths = []
    for ensemble in entries.ensemble.unique():
        tseries_path = os.path.join('tseries', tseries_fname(varname, component, experiment, ensemble))
        tseries_path_genlock = tseries_path + ".genlock"
        # if file doesn't exists and isn't being generated, generate it
        if clobber or (not os.path.exists(tseries_path) and not os.path.exists(tseries_path_genlock)):
            # create genlock file, indicating that tseries_path is being generated
            open(tseries_path_genlock, mode='w').close()
            # generate timeseries
            try:
                ds = _tseries_gen(varname, component, stream, experiment, ensemble, cluster_in)
            except:
                # error occured, remove genlock file and re-raise exception, to ease subsequent attempts
                os.remove(tseries_path_genlock)
                raise
            # write generated timeseries
            ds.to_netcdf(tseries_path, unlimited_dims=time_name, encoding={'region':{'dtype':'S1'}})
            # remove genlock file, indicating that tseries_path has been generated
            os.remove(tseries_path_genlock)
        # wait until genlock file doesn't exists, in case it was being generated or written
        while os.path.exists(tseries_path_genlock):
            print('genlock file exists, waiting')
            time.sleep(5)
        paths.append(tseries_path)

    # if there are multiple ensembles, concatenate over ensembles
    decode_times = True
    if len(paths) > 1:
        ds = xr.open_mfdataset(paths, decode_times=decode_times,
                               combine='nested', concat_dim='ensemble', data_vars=[varname])
        # force ensemble dimension to be last dimension
        # this make plotting in tseries_plot_1ds below more straightforward
        tb_name = ds.time.attrs['bounds']
        dims = list(ds[tb_name].dims)
        for dim in ds[varname].dims:
            if dim != 'ensemble' and dim != 'region' and dim not in dims:
                dims.append(dim)
        dims.extend(['region', 'ensemble'])
        ds = ds.transpose(*dims)
    else:
        ds = xr.open_dataset(paths[0], decode_times=decode_times)

    return ds.load()

def tseries_plot_1var(varname, ds_list, legend_list, title, figsize=(10, 6), region_val=None, vdim_name=None, vdim_ind=None, fname=None):
    """
    create a simple plot of a tseries variable for multiple datasets
    use units from last tseries variable for ylabel
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    for ds_ind, ds in enumerate(ds_list):
        t = time_year_plus_frac(ds, time_name)
        seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
        ax.plot(t, ds[varname].sel(seldict), label=legend_list[ds_ind])
    ax.set_xlabel('time (years)')
    ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    ax.set_title(title)
    if fname is not None:
        plt.savefig(fname)

def tseries_plot_1ds(ds, varnames, title, figsize=(10, 6), region_val=None, vdim_name=None, vdim_ind=None, fname=None):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    t = time_year_plus_frac(ds, time_name)
    seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
    for varname in varnames:
        if region_val is None:
            ax.plot(t, ds[varname], label=varname)
        else:
            ax.plot(t, ds[varname].sel(seldict), label=varname)
    ax.set_xlabel('time (years)')
    ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    ax.set_title(title)
    if fname is not None:
        plt.savefig(fname)

def tseries_plot_vars_vs_var(ds, varname_x, varnames_y, title, figsize=(10, 6), region_val=None, fname=None):
    """
    create a simple plot of a list of tseries variables vs a single tseries variable
    use units from last tseries variable for ylabel
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    for varname_y in varnames_y:
        if region_val is None:
            ax.plot(ds[varname_x], ds[varname_y], label=varname_y)
        else:
            ax.plot(ds[varname_x], ds[varname_y].sel(region=region_val), label=varname_y)
    ax.set_xlabel(varname_x + '(' + ds[varname_x].attrs['units'] + ')')
    ax.set_ylabel(ds[varname_y].attrs['units'])
    ax.legend()
    ax.set_title(title)
    if fname is not None:
        plt.savefig(fname)

def _seldict(ds, region_val, vdim_name, vdim_ind):
    """
    return dictionary of dimensions and indices
    to be used in sel operator
    """

    seldict = {}
    if region_val is not None:
        seldict['region'] = region_val
    if vdim_name is not None:
        if vdim_ind is None:
            vdim_ind_loc = -1 if vdim_name == 'lev' else 0
        else:
            vdim_ind_loc = vdim_ind
        seldict[vdim_name] = ds[vdim_name].values[vdim_ind_loc]

    return seldict

def _varname_resolved(varname, component):
    """resolve varname to underlying varname that appears in files"""

    with open(var_specs_fname, mode='r') as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname not in var_specs_all[component]['vars']:
        return varname
    
    var_spec = var_specs_all[component]['vars'][varname]
    
    return var_spec['varname'] if 'varname' in var_spec else varname

def _tseries_gen(varname, component, stream, experiment, ensemble, cluster_in):
    """
    generate a tseries for a particular ensemble member, return a Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    print('entering _tseries_gen for %s' % varname)
    varname_resolved = _varname_resolved(varname, component)
    fnames = data_catalog.get_files(
        variable=varname_resolved, component=component,
        stream=stream, experiment=experiment, ensemble=ensemble)
    print(fnames)

    with open(var_specs_fname, mode='r') as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname in var_specs_all[component]['vars']:
        var_spec = var_specs_all[component]['vars'][varname]
    else:
        var_spec = {}

    # use var specific reduce_dims if it exists, otherwise use reduce_dims for component
    if 'reduce_dims' in var_spec:
        reduce_dims = var_spec['reduce_dims']
    else:
        reduce_dims = var_specs_all[component]['reduce_dims']

    # get rank of varname from first file, used to set time chunksize
    # approximate number of time levels, assuming all files have same number
    # save time encoding from first file, to restore it in the multi-file case
    #     https://github.com/pydata/xarray/issues/2921
    with xr.open_dataset(fnames[0]) as ds_in:
        vardims = ds_in[varname_resolved].dims
        rank = len(vardims)
        vertlen = ds_in.dims[vardims[1]] if rank > 3 else 1
        tlen = ds_in.dims[time_name] * len(fnames)
        time_chunksize = 12 if rank < 4 else 1
        ds_in.chunk(chunks={time_name: time_chunksize})
        time_encoding = ds_in[time_name].encoding

    # instantiate cluster, if not provided via argument
    cluster = ncar_jobqueue.NCARCluster() if cluster_in is None else cluster_in

    workers = 2
    workers += 2 * math.log2(tlen / time_chunksize)
    workers += 2 * math.log2(vertlen)
    workers = 2 * round(workers/2) # round to nearest multiple of 2
    cluster.scale(workers)

    print(cluster.dashboard_link)

    # create dask distributed client, connecting to workers
    with dask.distributed.Client(cluster) as client:
        # tool to help track down file inconsistencies that trigger errors in open_mfdataset
        # test_open_mfdataset(fnames, time_chunksize)

        # data_vars = 'minimal', to avoid introducing time dimension to time-invariant fields when there are multiple files
        # only chunk in time, because if you chunk over spatial dims, then sum results depend on chunksize
        #     https://github.com/pydata/xarray/issues/2902
        with xr.open_mfdataset(fnames, data_vars='minimal', combine='by_coords',
                               chunks={time_name: time_chunksize}) as ds_in:
            # restore encoding for time from first file
            ds_in[time_name].encoding = time_encoding

            da_in = ds_in[varname_resolved]

            var_units = clean_units(da_in.attrs['units'])
            if 'unit_conv' in var_spec:
                var_units = '(%s)(%s)' % (str(var_spec['unit_conv']), var_units)

            # construct averaging/integrating weight
            weight = get_weight(ds_in, component, reduce_dims)
            weight_attrs = weight.attrs
            weight = get_rmask(ds_in, component) * weight
            weight.attrs = weight_attrs

            # use var specific tseries_op if it exists, otherwise use tseries_op for component
            if 'tseries_op' in var_spec:
                tseries_op = var_spec['tseries_op']
            else:
                tseries_op = var_specs_all[component]['tseries_op']

            if tseries_op == 'integrate':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                da_out.name = varname
                da_out.attrs['long_name'] = 'Integrated '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit('(%s)(%s)' % (weight.attrs['units'], var_units)).format()
            elif tseries_op == 'average':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                ones_masked = xr.ones_like(da_in).where(da_in.notnull())
                denom = (ones_masked * weight).sum(dim=reduce_dims)
                da_out /= denom
                da_out.name = varname
                da_out.attrs['long_name'] = 'Averaged '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit(var_units).format()
            else:
                msg = 'tseries_op==%s not implemented' % tseries_op
                raise NotImplementedError(msg)

            # force the computation to occur
            da_out.load()

            # propagate some settings from da_in to da_out
            da_out.encoding['dtype'] = da_in.encoding['dtype']
            copy_fill_settings(da_in, da_out)

            # change output units, if specified in var_spec
            units_key = 'integral_display_units' if tseries_op == 'integrate' else 'display_units'
            if units_key in var_spec:
                da_out.values = cf_units.Unit(da_out.attrs['units']).convert(
                    da_out.values, cf_units.Unit(clean_units(var_spec[units_key])))
                da_out.attrs['units'] = var_spec[units_key]

            ds_out = da_out.to_dataset()

            ds_out[time_name] = copy_fill_settings(ds_in[time_name], ds_out[time_name])

            # add time:bounds variable, if specified in ds_in
            if 'bounds' in ds_in[time_name].attrs:
                tb_name = ds_in[time_name].attrs['bounds']
                ds_out[tb_name] = ds_in[tb_name]

            # copy componet specific vars
            for copy_var_name in tseries_copy_var_names(component):
                copy_var_in = ds_in[copy_var_name]
                copy_var_out = copy_var_in
                ds_out[copy_var_name] = copy_fill_settings(copy_var_in, copy_var_out)

            # set ds_out.time to mid-interval values
            time_set_mid(ds_out, time_name)

            # copy file attributes
            ds_out.attrs = ds_in.attrs

            datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            ds_out.attrs['history'] = 'created by %s at %s' % (__file__, datestamp)

            ds_out.attrs['input_file_list'] = ' '.join(fnames)

    # if cluster was instantiated here, close it
    if cluster_in is None:
        cluster.close()

    return ds_out.load()

def test_open_mfdataset(paths, time_chunksize):
    for ind in range(len(paths)-1):
        print(' '.join(['testing open_mfdatset for', paths[ind], paths[ind+1]]))
        ds = xr.open_mfdataset(paths[ind:ind+2], data_vars='minimal', combine='by_coords',
                               chunks={time_name: time_chunksize})

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
    if component == 'ocn':
        dim_cnt_check(ds, 'dz', 1)
        dim_cnt_check(ds, 'TAREA', 2)
        volume = ds['dz'] * ds['TAREA']
        volume.attrs['units'] = 'cm3'
        return volume

    msg = 'get_volume not implemented for %s' % component
    raise NotImplementedError(msg)

def get_rmask(ds, component):
    """return region mask appropriate for component"""
    rmask_od = OrderedDict()
    if component == 'ocn':
        dim_cnt_check(ds, 'KMT', 2)
        lateral_dims = ds['KMT'].dims
        KMT = ds['KMT'].fillna(0)
        rmask_od['Global'] = xr.where(KMT > 0, 1.0, 0.0)
        rmask_od['SouOce (90S-30S)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(100.0) < -30.0), 1.0, 0.0)
        rmask_od['SH_high_lat (90S-44S)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(100.0) < -44.0), 1.0, 0.0)
        rmask_od['SH_mid_lat (44S-18S)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= -44.0) & (ds['TLAT'].fillna(100.0) < -18.0), 1.0, 0.0)
        rmask_od['low_lat (18S-18N)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= -18.0) & (ds['TLAT'].fillna(100.0) < 18.0), 1.0, 0.0)
        rmask_od['NH_mid_lat (18N-49N)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= 18.0) & (ds['TLAT'].fillna(100.0) < 49.0), 1.0, 0.0)
        rmask_od['NH_high_lat (49N-90N)'] = xr.where((KMT > 0) & (ds['TLAT'].fillna(-100.0) >= 49.0), 1.0, 0.0)
    if component == 'ice':
        dim_cnt_check(ds, 'tmask', 2)
        dim_cnt_check(ds, 'TLAT', 2)
        lateral_dims = ds['tmask'].dims
        rmask_od['NH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'].fillna(-100.0) >= 0.0), 1.0, 0.0)
        rmask_od['SH'] = xr.where((ds['tmask'] == 1) & (ds['TLAT'].fillna(100.0) < 0.0), 1.0, 0.0)
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

def tseries_copy_var_names(component):
    """return component specific list of vars to copy into generated tseries files"""
    if component == 'atm':
        return ['co2vmr', 'ch4vmr', 'f11vmr', 'f12vmr', 'n2ovmr', 'sol_tsi']
    return []
