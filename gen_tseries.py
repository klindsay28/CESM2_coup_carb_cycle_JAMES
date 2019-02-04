#! /usr/bin/env python

import argparse
from datetime import datetime, timezone
import math
import os

import xarray as xr
import yaml
from cf_units import Unit

import data_catalog
from tseries_utils import clean_units, get_weight, tseries_fname

parser = argparse.ArgumentParser(description="generate tseries files")
parser.add_argument(
    '--catalog_name', help='name of data_catalog to use', default='experiments')
parser.add_argument(
    '--tseries_specs_fname', help='name of tseries_spec file', required=True)
parser.add_argument(
    '--experiments', help='comma separated list of experiments to process', required=True)
parser.add_argument(
    '--time_chunk_cnt', help='size of dask chunks in time', type=int, default=1)

args = parser.parse_args()

with open(args.tseries_specs_fname, mode='r') as fptr:
    tseries_specs = yaml.load(fptr)

data_catalog.set_catalog(args.catalog_name)

for varname, ts_spec in tseries_specs.items():
    component = ts_spec['component']
    stream = ts_spec['stream']
    for experiment in args.experiments.split(','):
        print('varname=%s, experiment=%s' % (varname, experiment))
        entries = data_catalog.find_in_index(
            variable=varname, component=component, stream=stream, experiment=experiment)

        for ensemble in entries.ensemble.unique():
            fnames = data_catalog.get_files(
                variable=varname, component=component, stream=stream, experiment=experiment, ensemble=ensemble)
            print(fnames)

            with xr.open_mfdataset(fnames, decode_times=False, decode_coords=False) as ds_tmp:
                time_chunk_size = math.ceil(ds_tmp.dims['time'] / args.time_chunk_cnt)
                ds_in = ds_tmp.chunk({'time':time_chunk_size})
                print(ds_in.chunks)
                da_in = ds_in[varname]

                var_units = clean_units(da_in.attrs['units'])
                if 'unit_conv' in ts_spec:
                    var_units = '(%s)(%s)' % (str(ts_spec['unit_conv']), var_units)
                reduce_dims = ts_spec['reduce_dims']

                # construct averaging/integrating weight
                weight = get_weight(ds_in, component, reduce_dims)

                if ts_spec['weight_op'] == 'integrate':
                    da_out = (da_in * weight).sum(dim=reduce_dims)
                    da_out.name = varname
                    da_out.attrs['long_name'] = 'Integrated '+da_in.attrs['long_name']
                    da_out.attrs['units']=Unit('(%s)(%s)' % (weight.attrs['units'], var_units)).format()
                elif ts_spec['weight_op'] == 'average':
                    da_out = (da_in * weight).sum(dim=reduce_dims)
                    ones_masked = xr.ones_like(da_in).where(da_in.notnull())
                    denom = (ones_masked * weight).sum(dim=reduce_dims)
                    da_out /= denom
                    da_out.name = varname
                    da_out.attrs['long_name'] = 'Averaged '+da_in.attrs['long_name']
                    da_out.attrs['units']=Unit(var_units).format()
                else:
                    msg = 'weight_op==%s not implemented' % ts_spec['weight_op']
                    raise NotImplemented(msg)
                
                if 'units_out' in ts_spec:
                    da_out.values = Unit(da_out.attrs['units']).convert(
                        da_out.values, Unit(clean_units(ts_spec['units_out'])))
                    da_out.attrs['units'] = ts_spec['units_out']

                path = os.path.join('tseries', tseries_fname(varname, component, experiment, ensemble))
                merge_objs = []
                if 'bounds' in ds_in['time'].attrs:
                    tb = ds_in[ds_in['time'].attrs['bounds']]
                    tb.attrs['units'] = ds_in['time'].attrs['units']
                    tb.attrs['calendar'] = ds_in['time'].attrs['calendar']
                    merge_objs.append(tb)
                merge_objs.append(da_out)
                ds_out = xr.merge(merge_objs)
                ds_out.attrs = ds_in.attrs
                datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
                ds_out.attrs['history'] = 'created by %s at %s' % (__file__, datestamp)
                ds_out.to_netcdf(path, unlimited_dims='time')
                print('')
