"""interface for extracting timeseries from CESM output"""

from collections import OrderedDict
from datetime import datetime, timezone
import math
import os
import time
import warnings

import cf_units
import numpy as np
import xarray as xr
import yaml

import dask
# import dask_jobqueue
import ncar_jobqueue

import data_catalog
import esmlab_wrap
from utils import print_timestamp, copy_fill_settings, dim_cnt_check, time_set_mid, copy_var_names, drop_var_names
from utils_units import clean_units, conv_units
from CIME_shr_const import CIME_shr_const

from config import var_specs_fname
time_name = 'time'

def tseries_get_vars(varnames, component, experiment, stream=None, freq='mon', clobber=None, cluster_in=None):
    """
    return tseries for varnames, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called

    arguments are passed to tseries_get_var
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode='r') as fptr:
            var_specs_all = yaml.safe_load(fptr)
        stream = var_specs_all[component]['stream']

    if clobber is None:
        clobber = os.environ['CLOBBER'] == 'True' if 'CLOBBER' in os.environ else False

    # get matching data_catalog entries
    entries = data_catalog.find_in_index(
        variable=_varnames_resolved(varnames, component), component=component,
        stream=stream, experiment=experiment)

    # instantiate cluster, if not provided via argument
    # ignore dashboard warnings when instantiating
    if cluster_in is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module='.*dashboard')
            cluster = ncar_jobqueue.NCARCluster()
    else:
        cluster = cluster_in

    for varind, varname in enumerate(varnames):
        ds_tmp = tseries_get_var(varname, component, experiment, stream, freq, clobber, entries, cluster)
        if varind == 0:
            ds = ds_tmp
        else:
            ds[varname] = ds_tmp[varname]

    # if cluster was instantiated here, close it
    if cluster_in is None:
        ds.load()
        cluster.close()

    return ds

def tseries_get_var(varname, component, experiment, stream=None, freq='mon', clobber=None, entries_in=None, cluster_in=None):
    """
    return tseries for varname, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode='r') as fptr:
            var_specs_all = yaml.safe_load(fptr)
        stream = var_specs_all[component]['stream']

    if clobber is None:
        clobber = os.environ['CLOBBER'] == 'True' if 'CLOBBER' in os.environ else False

    # get matching data_catalog entries
    varname_resolved = _varname_resolved(varname, component)
    if entries_in is None:
        entries = data_catalog.find_in_index(
            variable=varname_resolved, component=component,
            stream=stream, experiment=experiment)
    else:
        entries = entries_in.loc[entries_in['variable'] == varname_resolved]

    if entries.empty:
        raise ValueError(f'no file matches found for varname={varname}, component={component}, experiment={experiment}')

    # loop over matching ensembles
    paths = []
    for ensemble in entries.ensemble.unique():
        path = _tseries_gen_wrap(varname, component, experiment, ensemble, freq, clobber, entries, cluster_in)
        paths.append(path)

    # if there are multiple ensembles, concatenate over ensembles
    decode_times = True
    if len(paths) > 1:
        ds = xr.open_mfdataset(paths, decode_times=decode_times,
                               combine='nested', concat_dim='ensemble', data_vars=[varname])
        # force ensemble dimension to be last dimension
        # this make plotting more straightforward
        tb_name = ds.time.attrs['bounds']
        dims = list(ds[tb_name].dims)
        for dim in ds.dims:
            if dim != 'ensemble' and dim != 'region' and dim not in dims:
                dims.append(dim)
        dims.extend(['region', 'ensemble'])
        ds = ds.transpose(*dims)
    else:
        ds = xr.open_dataset(paths[0], decode_times=decode_times)

    return ds

def _varnames_resolved(varnames, component):
    """resolve varnames to underlying varname that appears in files"""
    return [_varname_resolved(varname, component) for varname in varnames]

def _varname_resolved(varname, component):
    """resolve varname to underlying varname that appears in files"""

    with open(var_specs_fname, mode='r') as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname not in var_specs_all[component]['vars']:
        return varname

    var_spec = var_specs_all[component]['vars'][varname]

    return var_spec['varname'] if 'varname' in var_spec else varname

def _tseries_gen_wrap(varname, component, experiment, ensemble, freq, clobber, entries, cluster_in=None):
    """
    return path for file containing tseries for varname for a single ensemble member
        creating the file if necessary
    """
    if freq not in ['mon', 'ann']:
        msg = f'freq={freq} not implemented'
        raise NotImplementedError(msg)

    tseries_path = os.path.join('tseries', tseries_fname(varname, component, experiment, ensemble, freq))
    tseries_path_genlock = tseries_path + ".genlock"
    # if file doesn't exists and isn't being generated, generate it
    if clobber or (not os.path.exists(tseries_path) and not os.path.exists(tseries_path_genlock)):
        # create genlock file, indicating that tseries_path is being generated
        open(tseries_path_genlock, mode='w').close()
        # generate timeseries
        try:
            if freq == 'mon':
                ds = _tseries_gen(varname, component, ensemble, entries, cluster_in)
            if freq == 'ann':
                mon_path = _tseries_gen_wrap(varname, component, experiment,
                                             ensemble, 'mon', clobber, entries, cluster_in)
                print_timestamp(f'computing ann means from mon means for {varname}')
                ds = esmlab_wrap.compute_ann_mean(xr.open_dataset(mon_path))
        except:
            # error occured, remove genlock file and re-raise exception, to ease subsequent attempts
            os.remove(tseries_path_genlock)
            raise
        # write generated timeseries
        ds.to_netcdf(tseries_path, format='NETCDF4_CLASSIC')
        print_timestamp(f'{tseries_path} written')

        # remove genlock file, indicating that tseries_path has been generated
        os.remove(tseries_path_genlock)

    # wait until genlock file doesn't exists, in case it was being generated or written
    while os.path.exists(tseries_path_genlock):
        print_timestamp('genlock file exists, waiting')
        time.sleep(5)

    return tseries_path

def _tseries_gen(varname, component, ensemble, entries, cluster_in):
    """
    generate a tseries for a particular ensemble member, return a Dataset object
    """
    print_timestamp(f'entering _tseries_gen for {varname}')
    varname_resolved = _varname_resolved(varname, component)
    fnames = entries.loc[entries['ensemble'] == ensemble].files.tolist()
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
    with xr.open_dataset(fnames[0]) as ds0:
        vardims = ds0[varname_resolved].dims
        rank = len(vardims)
        vertlen = ds0.dims[vardims[1]] if rank > 3 else 1
        tlen = ds0.dims[time_name] * len(fnames)
        time_chunksize = 10*12 if rank < 4 else 2
        ds0.chunk(chunks={time_name: time_chunksize})
        time_encoding = ds0[time_name].encoding
        var_encoding = ds0[varname_resolved].encoding
        ds_encoding = ds0.encoding
        drop_var_names_loc = drop_var_names(component, ds0, varname_resolved)

    # instantiate cluster, if not provided via argument
    # ignore dashboard warnings when instantiating
    if cluster_in is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module='.*dashboard')
            cluster = ncar_jobqueue.NCARCluster()
    else:
        cluster = cluster_in

    workers = 1
    workers += math.log2(tlen / time_chunksize)
    workers += math.log2(vertlen)
    workers *= 4
    workers = 2 * round(workers/2) # round to nearest multiple of 2
    cluster.scale(workers)

    print(cluster.dashboard_link)

    # create dask distributed client, connecting to workers
    with dask.distributed.Client(cluster) as client:
        print_timestamp('client instantiated')

        # tool to help track down file inconsistencies that trigger errors in open_mfdataset
        # test_open_mfdataset(fnames, time_chunksize, varname)

        # data_vars = 'minimal', to avoid introducing time dimension to time-invariant fields when there are multiple files
        # only chunk in time, because if you chunk over spatial dims, then sum results depend on chunksize
        #     https://github.com/pydata/xarray/issues/2902
        with xr.open_mfdataset(fnames, data_vars='minimal', coords='minimal', compat='override', combine='by_coords',
                               chunks={time_name: time_chunksize}, drop_variables=drop_var_names_loc) as ds_in:
            print_timestamp('open_mfdataset returned')

            # restore encoding for time from first file
            ds_in[time_name].encoding = time_encoding

            da_in = ds_in[varname_resolved]
            da_in.encoding = var_encoding

            var_units = clean_units(da_in.attrs['units'])
            if 'unit_conv' in var_spec:
                var_units = f'({var_spec["unit_conv"]})({var_units})'

            # construct averaging/integrating weight
            weight = get_weight(ds_in, component, reduce_dims)
            weight_attrs = weight.attrs
            weight = get_rmask(ds_in, component) * weight
            weight.attrs = weight_attrs
            print_timestamp('weight constructed')

            # use var specific tseries_op if it exists, otherwise use tseries_op for component
            if 'tseries_op' in var_spec:
                tseries_op = var_spec['tseries_op']
            else:
                tseries_op = var_specs_all[component]['tseries_op']

            if tseries_op == 'integrate':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                da_out.name = varname
                da_out.attrs['long_name'] = 'Integrated '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit(f'({weight.attrs["units"]})({var_units})').format()
            elif tseries_op == 'average':
                da_out = (da_in * weight).sum(dim=reduce_dims)
                ones_masked = xr.ones_like(da_in).where(da_in.notnull())
                denom = (ones_masked * weight).sum(dim=reduce_dims)
                da_out /= denom
                da_out.name = varname
                da_out.attrs['long_name'] = 'Averaged '+da_in.attrs['long_name']
                da_out.attrs['units']=cf_units.Unit(var_units).format()
            else:
                msg = f'tseries_op={tseries_op} not implemented'
                raise NotImplementedError(msg)

            print_timestamp('da_out computation setup')

            # propagate some settings from da_in to da_out
            da_out.encoding['dtype'] = da_in.encoding['dtype']
            copy_fill_settings(da_in, da_out)

            # change output units, if specified in var_spec
            units_key = 'integral_display_units' if tseries_op == 'integrate' else 'display_units'
            if units_key in var_spec:
                conv_units(da_out, var_spec[units_key])
                print_timestamp('da_out units converted')

            ds_out = da_out.to_dataset()

            print_timestamp('ds_out generated')

            # add regional sum of weights
            weight_sum = weight.sum(dim=reduce_dims)
            weight_sum.attrs['long_name'] = 'sum of weights used in tseries generation'
            ds_out['weight_sum'] = weight_sum

            ds_out[time_name] = copy_fill_settings(ds_in[time_name], ds_out[time_name])

            # add time:bounds variable, if specified in ds_in
            if 'bounds' in ds_in[time_name].attrs:
                tb_name = ds_in[time_name].attrs['bounds']
                ds_out[tb_name] = ds_in[tb_name]

            # copy component specific vars
            for copy_var_name in copy_var_names(component):
                copy_var_in = ds_in[copy_var_name]
                copy_var_out = copy_var_in
                ds_out[copy_var_name] = copy_fill_settings(copy_var_in, copy_var_out)

            print_timestamp('copy_var_names added')

            # set ds_out.time to mid-interval values
            time_set_mid(ds_out, time_name)

            print_timestamp('time_set_mid returned')

            # copy file attributes
            ds_out.attrs = ds_in.attrs

            datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            msg = f'{datestamp}: created by {__file__}'
            if 'history' in ds_out.attrs:
                ds_out.attrs['history'] = '\n'.join([msg, ds_out.attrs['history']])
            else:
                ds_out.attrs['history'] = msg

            ds_out.attrs['input_file_list'] = ' '.join(fnames)

            for key in ['unlimited_dims']:
                if key in ds_encoding:
                    ds_out.encoding[key] = ds_encoding[key]

            # ensure NaN _FillValues do not get generated when the file is written out
            for var in ds_out.variables:
                if '_FillValue' not in ds_out[var].encoding:
                    ds_out[var].encoding['_FillValue'] = None

            # force computation of ds_out, while resources of client are still available
            ds_out.load()

    print_timestamp('ds_in and client closed')

    # if cluster was instantiated here, close it
    if cluster_in is None:
        cluster.close()

    return ds_out

def test_open_mfdataset(paths, time_chunksize, varname=None):
    for ind in range(len(paths)-1):
        print(' '.join(['testing open_mfdatset for', paths[ind], paths[ind+1]]))
        ds = xr.open_mfdataset(paths[ind:ind+2], data_vars='minimal', combine='by_coords',
                               chunks={time_name: time_chunksize})
        if varname is not None:
            print(ds[varname])

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
    msg = f'unknown component={component}'
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
        area_earth = 4.0 * CIME_shr_const('pi') * CIME_shr_const('rearth')**2 # area of earth in CIME [m2]

        # normalize area so that sum over 'lat', 'lon' yields area_earth
        area = ds['gw'] + 0.0 * ds['lon'] # add 'lon' dimension
        area = (area_earth / area.sum(dim=('lat', 'lon'))) * area
        area.attrs['units'] = 'm2'
        return area
    msg = f'unknown component={component}'
    raise ValueError(msg)

def get_volume(ds, component):
    """return volume DataArray appropriate for component"""
    if component == 'ocn':
        dim_cnt_check(ds, 'dz', 1)
        dim_cnt_check(ds, 'TAREA', 2)
        volume = ds['dz'] * ds['TAREA']
        volume.attrs['units'] = 'cm3'
        return volume

    msg = f'component={component} not implemented'
    raise NotImplementedError(msg)

def get_rmask(ds, component):
    """return region mask appropriate for component"""
    rmask_od = OrderedDict()
    if component == 'ocn':
        dim_cnt_check(ds, 'KMT', 2)
        lateral_dims = ds['KMT'].dims
        KMT = ds['KMT'].fillna(0).load() # treat missing-values, which arise from land-block elimination as 0
        TLAT = ds['TLAT'].load()
        rmask_od['Global'] = xr.where(KMT > 0, 1.0, 0.0)
        rmask_od['SouOce (90S-30S)'] = xr.where((KMT > 0) & (TLAT < -30.0), 1.0, 0.0)
        rmask_od['SH_high_lat (90S-44S)'] = xr.where((KMT > 0) & (TLAT < -44.0), 1.0, 0.0)
        rmask_od['SH_mid_lat (44S-18S)'] = xr.where((KMT > 0) & (TLAT >= -44.0) & (TLAT < -18.0), 1.0, 0.0)
        rmask_od['low_lat (18S-18N)'] = xr.where((KMT > 0) & (TLAT >= -18.0) & (TLAT < 18.0), 1.0, 0.0)
        rmask_od['NH_mid_lat (18N-49N)'] = xr.where((KMT > 0) & (TLAT >= 18.0) & (TLAT < 49.0), 1.0, 0.0)
        rmask_od['NH_high_lat (49N-90N)'] = xr.where((KMT > 0) & (TLAT >= 49.0), 1.0, 0.0)
    if component == 'ice':
        dim_cnt_check(ds, 'tmask', 2)
        dim_cnt_check(ds, 'TLAT', 2)
        lateral_dims = ds['tmask'].dims
        tmask = ds['tmask'].load()
        TLAT = ds['TLAT'].load()
        rmask_od['NH'] = xr.where((tmask == 1) & (TLAT >= 0.0), 1.0, 0.0)
        rmask_od['SH'] = xr.where((tmask == 1) & (TLAT < 0.0), 1.0, 0.0)
    if component == 'lnd':
        dim_cnt_check(ds, 'landfrac', 2)
        lateral_dims = ds['landfrac'].dims
        rmask_od['Global'] = xr.where(ds['landfrac'] > 0, 1.0, 0.0)
#         lat = ds['lat'].load()
#         lon = ds['lon'].load()
#         rmask_od['CentralAfrica'] = xr.where((ds['landfrac'] > 0)
#                                              & (lat >= -5.0) & (lat < 5.0)
#                                              & (lon >= 0.0) & (lon < 30.0), 1.0, 0.0)
    if component == 'atm':
        dim_cnt_check(ds, 'gw', 1)
        lateral_dims = ('lat', 'lon')
        lat = ds['lat'].load()
        lon = ds['lon'].load()
        rmask_od['Global'] = xr.where((lat > -100.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od['SH'] = xr.where((lat < 0.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od['SH_Trop'] = xr.where((lat > -30) & (lat < 0.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od['NH'] = xr.where((lat > 0.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od['NH_Trop'] = xr.where((lat > 0.0) & (lat < 30.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od['nino34'] = xr.where((lat > -5.0) & (lat < 5.0) & (lon > 190) & (lon < 240), 1.0, 0.0)
    if len(rmask_od) == 0:
        msg = f'unknown component={component}'
        raise ValueError(msg)

    print_timestamp('rmask_od created')

    rmask = xr.DataArray(np.zeros((len(rmask_od), ds.dims[lateral_dims[0]], ds.dims[lateral_dims[1]])),
                         dims=('region', lateral_dims[0], lateral_dims[1]),
                         coords={'region':list(rmask_od.keys())})
    rmask.region.encoding['dtype'] = 'S1'

    # add coordinates if appropriate
    if component == 'atm' or component == 'lnd':
        rmask.coords['lat'] = ds['lat']
        rmask.coords['lon'] = ds['lon']

    for i, rmask_field in enumerate(rmask_od.values()):
        rmask.values[i,:,:] = rmask_field

    return rmask

def tseries_fname(varname, component, experiment, ensemble, freq):
    """return relative filename for tseries"""
    return f'{varname}_{component}_{experiment}_{ensemble:02d}_{freq}.nc'
