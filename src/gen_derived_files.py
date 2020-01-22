"""generate single variable timeseries files of derived fields"""

import os.path
import subprocess
import yaml

import xarray as xr

from src import data_catalog
from src.config import var_specs_fname, expr_metadata_fname
from src.utils import copy_fill_settings

def main():
    data_catalog.set_catalog('experiments')

    component = 'ocn'

    experiments = ['piControl-cmip5', 'historical-cmip5', 'rcp85-cmip5',
                   'esm-piControl-cmip5', 'esm-hist-cmip5', 'esm-rcp85-cmip5']

    varnames = ['photoC_diat_zint_100m', 'photoC_sp_zint_100m', 'photoC_diaz_zint_100m',
                'photoC_diat_zint', 'photoC_sp_zint', 'photoC_diaz_zint',
                'POC_FLUX_100m', 'CaCO3_FLUX_100m']

    for experiment in experiments:
        for varname in varnames:
            gen_derived_files(varname, component, experiment)

def gen_derived_files(varname, component, experiment, stream=None, keep_read_src_files=True):
    """
    assumes that data_catalog.set_catalog has been called
    """

    with open(var_specs_fname, mode='r') as fptr:
        var_specs_all = yaml.safe_load(fptr)
    var_spec = var_specs_all[component]['vars'][varname]

    # if this varname is not derived from others, there is nothing to be done
    if 'derived_from_varnames' not in var_spec:
        return

    # basic error checking on specified variable derivation
    if var_spec['derive_op'] not in ['zint_100m', 'zint', 'sel_100m']:
        raise ValueError(f'unknown derive_op={var_spec["derive_op"]} for {varname}')
    if var_spec['derive_op'] in ['zint_100m', 'zint', 'sel_100m']:
        if len(var_spec['derived_from_varnames']) > 1:
            raise ValueError(f'too many derived_from_varnames specified for derive_op={var_spec["derive_op"]} for {varname}')

    # if no stream is specified, get the default stream for this component
    if stream is None:
        stream = var_specs_all[component]['stream']

    with open(expr_metadata_fname, mode='r') as fptr:
        expr_metadata = yaml.safe_load(fptr)['experiments']['data_sources'][experiment]

    # loop over matching ensembles
    for ensemble in range(len(expr_metadata)):
        # generate HPSS dir listing of all tseries files for component in this experiment
        hpss_dir = '/'.join([expr_metadata[ensemble]['hpss_root_dir'], component, 'proc', 'tseries'])
        hsi_cmd = f'hsi -P ls -R -1 {hpss_dir}'.split(' ')
        hsi_out = subprocess.check_output(hsi_cmd).decode().split('\n')

        # generate lists of filenames for each source variable
        src_fnames_hpss_dict = {}
        for src_varname in var_spec['derived_from_varnames']:
            str_match = '.'.join([stream, src_varname, ''])
            src_fnames_hpss_dict[src_varname] = [fname for fname in hsi_out if str_match in fname]

        # ensure that src_fnames_hpss_dict for all source variables have the same date strings
        src_varname_0 = var_spec['derived_from_varnames'][0]
        src_fnames_datestrs_0 = [fname.split('.')[-2] for fname in src_fnames_hpss_dict[src_varname_0]]
        for src_varname in var_spec['derived_from_varnames'][1:]:
            src_fnames_datestrs = [fname.split('.')[-2] for fname in src_fnames_hpss_dict[src_varname]]
            if src_fnames_datestrs != src_fnames_datestrs_0:
                print(f'filename date-strings for {src_varname_0} do not match those for {src_varname}')
                print(src_fnames_hpss_dict[src_varname_0])
                print(src_fnames_hpss_dict[src_varname])
                print(src_fnames_datestrs_0)
                print(src_fnames_datestrs)
                raise ValueError()

        # generate filenames for varname, from those for src_varname_0
        dir_old = expr_metadata[ensemble]['hpss_root_dir']
        dir_new = expr_metadata[ensemble]['root_dir']
        src_fnames_disk = [fname.replace(dir_old, dir_new) for fname in src_fnames_hpss_dict[src_varname_0]]
        var_old = '.'.join(['', src_varname_0, ''])
        var_new = '.'.join(['', varname, ''])
        dst_fnames_disk = [fname.replace(var_old, var_new) for fname in src_fnames_disk]

        src_fnames_disk_read = []

        # generate dst files that do not already exist
        for fname_ind, dst_fname_disk in enumerate(dst_fnames_disk):
            if os.path.isfile(dst_fname_disk):
                continue
            print(f'generating {dst_fname_disk}')
            src_fname_disk_dict = {}
            for src_varname in var_spec['derived_from_varnames']:
                src_fname_hpss = src_fnames_hpss_dict[src_varname][fname_ind]
                src_fname_disk = src_fname_hpss.replace(dir_old, dir_new)
                if not os.path.isfile(src_fname_disk):
                    hsi_cmd = f'hsi -P get {src_fname_disk} : {src_fname_hpss}'.split(' ')
                    hsi_out = subprocess.check_output(hsi_cmd).decode().split('\n')
                    print(hsi_out)
                    src_fnames_disk_read.append(src_fname_disk)
                src_fname_disk_dict[src_varname] = src_fname_disk
            if var_spec['derive_op'] in ['zint_100m', 'zint', 'sel_100m']:
                gen_file_z_reduce(var_spec['derive_op'], src_fname_disk_dict, varname, dst_fname_disk)

    return

def gen_file_z_reduce(z_reduce_op, src_fname_disk_dict, varname, dst_fname_disk):
    src_varname = list(src_fname_disk_dict)[0]
    src_fname_disk = src_fname_disk_dict[src_varname]
    with xr.open_dataset(src_fname_disk) as ds_in:
        vert_dims = ['z_t', 'z_t_150m', 'z_w', 'z_w_bot', 'z_w_top', 'moc_z']
        drop_varnames = [varname_tmp for varname_tmp in ds_in.variables if set(vert_dims) & set(ds_in[varname_tmp].dims)]

        # drop non-used coordinates and variables that use them
        # their presence causes xarray to generate incorrect coordinate attributes
        coordnames = ['TLAT', 'TLONG', 'ULAT', 'ULONG']
        drop_coordnames = [coordname for coordname in coordnames if coordname not in ds_in[src_varname].encoding['coordinates']]
        drop_varnames.extend(drop_coordnames)
        vars_on_drop_coordnames = [varname_tmp for varname_tmp in ds_in.variables
                                   if 'coordinates' in ds_in[varname_tmp].encoding
                                   and (set(drop_coordnames) & set(ds_in[varname_tmp].encoding['coordinates'].split(' ')))]
        drop_varnames.extend(vars_on_drop_coordnames)

        ds_out = ds_in.drop(drop_varnames)
        for varname_tmp in ds_out.variables:
            copy_fill_settings(ds_in[varname_tmp], ds_out[varname_tmp])

        vert_dim = ds_in[src_varname].dims[1]
        # initial setup of generated DataArray, preserves metadata (coordinates, attributes, encoding)
        da_out = ds_in.drop(drop_coordnames)[src_varname].isel({vert_dim: 0}).drop(vert_dim)

        # set vertical reduction metadata
        da_out.attrs['grid_loc'] = ''.join(['2', da_out.attrs['grid_loc'][1:3], '0'])

        # compute reduction on vert_dim

        if z_reduce_op in ['zint_100m', 'zint']:
            if vert_dim not in ['z_t', 'z_t_150m']:
                raise NotImplementedError(f'vert_dim={vert_dim} not implemented')

            vert_dim_len = ds_in.dims[vert_dim]
            zint_weight = ds_in['dz'].isel(z_t=slice(0, vert_dim_len)).rename({'z_t': vert_dim})
            sel_dict = {}
            if z_reduce_op == 'zint_100m':
                sel_dict[vert_dim] = slice(0, 100.0e2)
            da_out.values = (zint_weight.sel(sel_dict) * ds_in[src_varname].sel(sel_dict)).sum(vert_dim)

            # set zint_100m, zint specific metadata
            da_out.attrs['units'] = ' '.join([da_out.attrs['units'], 'cm'])
            long_name = da_out.attrs['long_name']
            long_name += ' Vertical Integral'
            if z_reduce_op == 'zint_100m':
                long_name_suffix += ', 0-100m'
            da_out.attrs['long_name'] = long_name

        if z_reduce_op == 'sel_100m':
            sel_dict = {vert_dim: 100.0e2}
            da_out.values = ds_in[src_varname].sel(sel_dict, method='nearest')

            # set sel_100m specific metadata
            long_name = da_out.attrs['long_name']
            long_name += ' at 100m'
            da_out.attrs['long_name'] = long_name

        ds_out[varname] = da_out

        ds_out.to_netcdf(dst_fname_disk)

        # append, using NCO, dropped coordinates and variables that use them
        vars = ','.join(drop_coordnames + vars_on_drop_coordnames)
        cmd = ['ncks', '-A', '-v', vars, src_fname_disk, dst_fname_disk]
        cmd_out = subprocess.check_output(cmd).decode().split('\n')

    return

if __name__ == '__main__':
    main()
