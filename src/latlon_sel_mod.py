"""interface for extracting variables at a lat-lon location from CESM output"""

from datetime import datetime, timezone
import math
import os
import sys
import time
import warnings

import cf_units
import xarray as xr
import yaml

from src import data_catalog
from src import esmlab_wrap
from src.utils import (
    print_timestamp,
    time_set_mid,
    copy_var_names,
    drop_var_names,
)
from src.utils_grid import get_latlon_isel_dict
from src.config import rootdir, var_specs_fname

time_name = "time"

cache_dir_default = os.path.join(rootdir, "latlon_sel")


def latlon_sel_get_var(
    varname,
    lat,
    lon,
    component,
    experiment,
    stream=None,
    cache_dir=cache_dir_default,
    clobber=None,
    entries_in=None,
):
    """
    return values for varname, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode="r") as fptr:
            var_specs_all = yaml.safe_load(fptr)
        stream_loc = var_specs_all[component]["stream"]
    else:
        stream_loc = stream

    if clobber is None:
        clobber = os.environ["CLOBBER"] == "True" if "CLOBBER" in os.environ else False

    # get matching data_catalog entries
    varname_resolved = _varname_resolved(varname, component)
    if entries_in is None:
        entries = data_catalog.find_in_index(
            variable=varname_resolved,
            component=component,
            stream=stream_loc,
            experiment=experiment,
        )
    else:
        entries = entries_in.loc[entries_in["variable"] == varname_resolved]

    if entries.empty:
        raise ValueError(
            f"no file matches found for varname={varname}, component={component}, experiment={experiment}"
        )

    # loop over matching ensembles
    paths = []
    for ensemble in entries.ensemble.unique():
        path = _latlon_sel_gen_wrap(
            varname,
            lat,
            lon,
            component,
            experiment,
            ensemble,
            cache_dir,
            clobber,
            entries,
        )
        paths.append(path)

    # if there are multiple ensembles, concatenate over ensembles
    decode_times = True
    if len(paths) > 1:
        ds = xr.open_mfdataset(
            paths,
            decode_times=decode_times,
            combine="nested",
            concat_dim="ensemble",
            data_vars=[varname],
        )
    else:
        ds = xr.open_dataset(paths[0], decode_times=decode_times)

    return ds


def _varname_resolved(varname, component):
    """resolve varname to underlying varname that appears in files"""

    with open(var_specs_fname, mode="r") as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname not in var_specs_all[component]["vars"]:
        return varname

    var_spec = var_specs_all[component]["vars"][varname]

    return var_spec["varname"] if "varname" in var_spec else varname


def _latlon_sel_gen_wrap(
    varname, lat, lon, component, experiment, ensemble, cache_dir, clobber, entries,
):
    """
    return path for file containing values for varname for a single ensemble member
        creating the file if necessary
    """

    fnames = entries.files.tolist()
    with xr.open_dataset(fnames[0]) as ds0:
        isel_dict = get_latlon_isel_dict(ds0, component, lat, lon)
    cache_path = os.path.join(
        cache_dir, latlon_sel_fname(varname, isel_dict, component, experiment, ensemble)
    )
    cache_path_genlock = cache_path + ".genlock"
    # if file doesn't exists and isn't being generated, generate it
    if clobber or (
        not os.path.exists(cache_path) and not os.path.exists(cache_path_genlock)
    ):
        # create genlock file, indicating that cache_path is being generated
        open(cache_path_genlock, mode="w").close()
        # generate timeseries
        try:
            ds = _latlon_sel_gen(varname, isel_dict, component, ensemble, entries)
        except:
            # error occured, remove genlock file and re-raise exception, to ease subsequent attempts
            os.remove(cache_path_genlock)
            raise

        # write generated timeseries
        # ensure NaN _FillValues do not get generated
        for var in ds.variables:
            if "_FillValue" not in ds[var].encoding:
                ds[var].encoding["_FillValue"] = None
        # remove attributes with forbidden names
        for att_name in ["_NCProperties"]:
            if att_name in ds.attrs:
                del ds.attrs[att_name]
        ds.to_netcdf(cache_path, format="NETCDF4_CLASSIC")
        print_timestamp(f"{cache_path} written")

        # remove genlock file, indicating that cache_path has been generated
        os.remove(cache_path_genlock)

    # wait until genlock file doesn't exists, in case it was being generated or written
    while os.path.exists(cache_path_genlock):
        print_timestamp("genlock file exists, waiting")
        time.sleep(5)

    return cache_path


def _latlon_sel_gen(varname, isel_dict, component, ensemble, entries):
    """
    generate a values for a particular ensemble member, return a Dataset object
    """
    print_timestamp(f"varname={varname}")
    varname_resolved = _varname_resolved(varname, component)
    fnames = entries.loc[entries["ensemble"] == ensemble].files.tolist()
    print(fnames)

    with open(var_specs_fname, mode="r") as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname in var_specs_all[component]["vars"]:
        var_spec = var_specs_all[component]["vars"][varname]
    else:
        var_spec = {}

    ds_out_list = []

    with xr.open_dataset(fnames[0]) as ds0:
        drop_var_names_loc = drop_var_names(component, ds0, varname_resolved)
        var_list = [time_name, varname]
        if "bounds" in ds0[time_name].attrs:
            var_list.append(ds0[time_name].attrs["bounds"])
        var_list.extend(copy_var_names(component))
        for fname in fnames:
            with xr.open_dataset(fname, drop_variables=drop_var_names_loc) as ds_in:
                ds_out = ds_in[var_list].isel(isel_dict)
                ds_out_list.append(ds_out.load())

        ds_out = xr.concat(
            ds_out_list, dim=time_name, coords="minimal", compat="override"
        )

        # restore encoding for time from first file
        ds_out[time_name].encoding = ds0[time_name].encoding

        # set ds_out.time to mid-interval values
        ds_out = time_set_mid(ds_out, time_name)

        # copy file attributes
        ds_out.attrs = ds0.attrs

    datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    msg = f"{datestamp}: created by {__file__}"
    if "history" in ds_out.attrs:
        ds_out.attrs["history"] = "\n".join([msg, ds_out.attrs["history"]])
    else:
        ds_out.attrs["history"] = msg

    ds_out.attrs["input_file_list"] = " ".join(fnames)

    return ds_out


def latlon_sel_fname(varname, isel_dict, component, experiment, ensemble):
    """
    return relative filename for latlon_sel

    filename is based on an isel_dict instead of a sel_dict so that lat,lon values
    that map to the same grid cell map to the same filename
    """
    isel_dict_str = "_".join([f"{key}_{val}" for key, val in isel_dict.items()])
    return f"{varname}_{isel_dict_str}_{component}_{experiment}_{ensemble:02d}.nc"
