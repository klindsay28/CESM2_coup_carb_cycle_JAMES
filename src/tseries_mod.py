"""interface for extracting timeseries from CESM output"""

from datetime import datetime, timezone
import math
import os
import sys
import time
import warnings

import cf_units
import xarray as xr
import yaml

import dask

# import dask_jobqueue
try:
    import ncar_jobqueue
except RuntimeError:
    pass

from src import data_catalog
from src import esmlab_wrap
from src.utils import (
    print_timestamp,
    copy_fill_settings,
    time_set_mid,
    copy_var_names,
    drop_var_names,
)
from src.utils_grid import get_weight, get_rmask
from src.utils_units import clean_units, conv_units
from src.config import rootdir, var_specs_fname

time_name = "time"

cache_dir_default = os.path.join(rootdir, "tseries")


def tseries_get_vars(
    varnames,
    component,
    experiment,
    stream=None,
    freq="mon",
    cache_dir=cache_dir_default,
    clobber=None,
    cluster_in=None,
):
    """
    return tseries for varnames, as a xarray.Dataset object
    assumes that data_catalog.set_catalog has been called

    arguments are passed to tseries_get_var
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
    entries = data_catalog.find_in_index(
        variable=_varnames_resolved(varnames, component),
        component=component,
        stream=stream_loc,
        experiment=experiment,
    )

    # instantiate cluster, if not provided via argument
    # ignore dashboard warnings when instantiating
    if cluster_in is None and "ncar_jobqueue" in sys.modules:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", module=".*dashboard")
            cluster = ncar_jobqueue.NCARCluster()
    else:
        cluster = cluster_in

    ds = xr.merge(
        [
            tseries_get_var(
                varname,
                component,
                experiment,
                stream_loc,
                freq,
                cache_dir,
                clobber,
                entries,
                cluster,
            )
            for varname in varnames
        ]
    )

    # if cluster was instantiated here, close it
    if cluster_in is None and "ncar_jobqueue" in sys.modules:
        ds.load()
        cluster.close()

    return ds


def tseries_get_var(
    varname,
    component,
    experiment,
    stream=None,
    freq="mon",
    cache_dir=cache_dir_default,
    clobber=None,
    entries_in=None,
    cluster_in=None,
):
    """
    return tseries for varname, as a xarray.Dataset object
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
        path = _tseries_gen_wrap(
            varname,
            component,
            experiment,
            ensemble,
            freq,
            cache_dir,
            clobber,
            entries,
            cluster_in,
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


def _varnames_resolved(varnames, component):
    """resolve varnames to underlying varname that appears in files"""
    return [_varname_resolved(varname, component) for varname in varnames]


def _varname_resolved(varname, component):
    """resolve varname to underlying varname that appears in files"""

    with open(var_specs_fname, mode="r") as fptr:
        var_specs_all = yaml.safe_load(fptr)

    if varname not in var_specs_all[component]["vars"]:
        return varname

    var_spec = var_specs_all[component]["vars"][varname]

    return var_spec["varname"] if "varname" in var_spec else varname


def _tseries_gen_wrap(
    varname,
    component,
    experiment,
    ensemble,
    freq,
    cache_dir,
    clobber,
    entries,
    cluster_in=None,
):
    """
    return path for file containing tseries for varname for a single ensemble member
        creating the file if necessary
    """
    if freq not in ["mon", "ann"]:
        msg = f"freq={freq} not implemented"
        raise NotImplementedError(msg)

    tseries_path = os.path.join(
        cache_dir, tseries_fname(varname, component, experiment, ensemble, freq)
    )
    tseries_path_genlock = tseries_path + ".genlock"
    # if file doesn't exists and isn't being generated, generate it
    if clobber or (
        not os.path.exists(tseries_path) and not os.path.exists(tseries_path_genlock)
    ):
        # create genlock file, indicating that tseries_path is being generated
        open(tseries_path_genlock, mode="w").close()
        # generate timeseries
        try:
            if freq == "mon":
                ds = _tseries_gen(varname, component, ensemble, entries, cluster_in)
            if freq == "ann":
                mon_path = _tseries_gen_wrap(
                    varname,
                    component,
                    experiment,
                    ensemble,
                    "mon",
                    cache_dir,
                    clobber,
                    entries,
                    cluster_in,
                )
                print_timestamp(f"computing ann means from mon means for {varname}")
                ds = esmlab_wrap.compute_ann_mean(xr.open_dataset(mon_path))
        except:
            # error occured, remove genlock file and re-raise exception, to ease subsequent attempts
            os.remove(tseries_path_genlock)
            raise

        # write generated timeseries
        # ensure NaN _FillValues do not get generated
        for var in ds.variables:
            if "_FillValue" not in ds[var].encoding:
                ds[var].encoding["_FillValue"] = None
        ds.to_netcdf(tseries_path, format="NETCDF4_CLASSIC")
        print_timestamp(f"{tseries_path} written")

        # remove genlock file, indicating that tseries_path has been generated
        os.remove(tseries_path_genlock)

    # wait until genlock file doesn't exists, in case it was being generated or written
    while os.path.exists(tseries_path_genlock):
        print_timestamp("genlock file exists, waiting")
        time.sleep(5)

    return tseries_path


def _tseries_gen(varname, component, ensemble, entries, cluster_in):
    """
    generate a tseries for a particular ensemble member, return a Dataset object
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

    # use var specific reduce_dims if it exists, otherwise use reduce_dims for component
    if "reduce_dims" in var_spec:
        reduce_dims = var_spec["reduce_dims"]
    else:
        reduce_dims = var_specs_all[component]["reduce_dims"]

    # get rank of varname from first file, used to set time chunksize
    # approximate number of time levels, assuming all files have same number
    # save time encoding from first file, to restore it in the multi-file case
    #     https://github.com/pydata/xarray/issues/2921
    with xr.open_dataset(fnames[0]) as ds0:
        vardims = ds0[varname_resolved].dims
        rank = len(vardims)
        vertlen = ds0.dims[vardims[1]] if rank > 3 else 0
        time_chunksize = 10 * 12 if rank < 4 else 6
        ds0.chunk(chunks={time_name: time_chunksize})
        time_encoding = ds0[time_name].encoding
        var_encoding = ds0[varname_resolved].encoding
        ds0_attrs = ds0.attrs
        ds0_encoding = ds0.encoding
        drop_var_names_loc = drop_var_names(component, ds0, varname_resolved)

    # instantiate cluster, if not provided via argument
    # ignore dashboard warnings when instantiating
    if cluster_in is None:
        if "ncar_jobqueue" in sys.modules:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", module=".*dashboard")
                cluster = ncar_jobqueue.NCARCluster()
        else:
            raise ValueError(
                "cluster_in not provided and ncar_jobqueue did not load successfully"
            )
    else:
        cluster = cluster_in

    workers = 12
    if vertlen >= 20:
        workers *= 2
    if vertlen >= 60:
        workers *= 2
    workers = 2 * round(workers / 2)  # round to nearest multiple of 2
    print_timestamp(f"calling cluster.scale({workers})")
    cluster.scale(workers)

    print_timestamp(f"dashboard_link={cluster.dashboard_link}")

    # create dask distributed client, connecting to workers
    with dask.distributed.Client(cluster) as client:
        print_timestamp("client instantiated")

        # tool to help track down file inconsistencies that trigger errors in open_mfdataset
        # test_open_mfdataset(fnames, time_chunksize, varname)

        # data_vars = "minimal", to avoid introducing time dimension to time-invariant fields when there are multiple files
        # only chunk in time, because if you chunk over spatial dims, then sum results depend on chunksize
        #     https://github.com/pydata/xarray/issues/2902
        with xr.open_mfdataset(
            fnames,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine="by_coords",
            chunks={time_name: time_chunksize},
            drop_variables=drop_var_names_loc,
        ) as ds_in:
            print_timestamp("open_mfdataset returned")

            # restore encoding for time from first file
            ds_in[time_name].encoding = time_encoding

            da_in_full = ds_in[varname_resolved]
            da_in_full.encoding = var_encoding

            var_units = clean_units(da_in_full.attrs["units"])
            if "unit_conv" in var_spec:
                var_units = f'({var_spec["unit_conv"]})({var_units})'

            # construct averaging/integrating weight
            weight = get_weight(ds_in, component, reduce_dims)
            weight_attrs = weight.attrs
            weight = get_rmask(ds_in, component) * weight
            weight.attrs = weight_attrs
            print_timestamp("weight constructed")

            # compute regional sum of weights
            da_in_t0 = da_in_full.isel({time_name: 0}).drop(time_name)
            ones_masked_t0 = xr.ones_like(da_in_t0).where(da_in_t0.notnull())
            weight_sum = (ones_masked_t0 * weight).sum(dim=reduce_dims)
            weight_sum.name = f"weight_sum_{varname}"
            weight_sum.attrs = weight.attrs
            weight_sum.attrs[
                "long_name"
            ] = f"sum of weights used in tseries generation for {varname}"

            tlen = da_in_full.sizes[time_name]
            print_timestamp(f"tlen={tlen}")

            # use var specific tseries_op if it exists, otherwise use tseries_op for component
            if "tseries_op" in var_spec:
                tseries_op = var_spec["tseries_op"]
            else:
                tseries_op = var_specs_all[component]["tseries_op"]

            ds_out_list = []

            time_step_nominal = min(2 * workers * time_chunksize, tlen)
            time_step = math.ceil(tlen / (tlen // time_step_nominal))
            print_timestamp(f"time_step={time_step}")
            for time_ind0 in range(0, tlen, time_step):
                print_timestamp(f"time_ind={time_ind0}, {time_ind0 + time_step}")
                da_in = da_in_full.isel(
                    {time_name: slice(time_ind0, time_ind0 + time_step)}
                )

                if tseries_op == "integrate":
                    da_out = (da_in * weight).sum(dim=reduce_dims)
                    da_out.name = varname
                    da_out.attrs["long_name"] = "Integrated " + da_in.attrs["long_name"]
                    da_out.attrs["units"] = cf_units.Unit(
                        f'({weight.attrs["units"]})({var_units})'
                    ).format()
                elif tseries_op == "average":
                    da_out = (da_in * weight).sum(dim=reduce_dims)
                    ones_masked = xr.ones_like(da_in).where(da_in.notnull())
                    denom = (ones_masked * weight).sum(dim=reduce_dims)
                    da_out /= denom
                    da_out.name = varname
                    da_out.attrs["long_name"] = "Averaged " + da_in.attrs["long_name"]
                    da_out.attrs["units"] = cf_units.Unit(var_units).format()
                else:
                    msg = f"tseries_op={tseries_op} not implemented"
                    raise NotImplementedError(msg)

                print_timestamp("da_out computation setup")

                # propagate some settings from da_in to da_out
                da_out.encoding["dtype"] = da_in.encoding["dtype"]
                copy_fill_settings(da_in, da_out)

                ds_out = da_out.to_dataset()

                print_timestamp("ds_out generated")

                # copy particular variables from ds_in
                copy_var_list = [time_name]
                if "bounds" in ds_in[time_name].attrs:
                    copy_var_list.append(ds_in[time_name].attrs["bounds"])
                copy_var_list.extend(copy_var_names(component))
                ds_out = xr.merge(
                    [
                        ds_out,
                        ds_in[copy_var_list].isel(
                            {time_name: slice(time_ind0, time_ind0 + time_step)}
                        ),
                    ]
                )

                print_timestamp("copy_var_names added")

                # force computation of ds_out, while resources of client are still available
                print_timestamp("calling ds_out.load")
                ds_out_list.append(ds_out.load())
                print_timestamp("returned from ds_out.load")

            print_timestamp("concatenating ds_out_list datasets")
            ds_out = xr.concat(
                ds_out_list,
                dim=time_name,
                data_vars=[varname],
                coords="minimal",
                compat="override",
            )

            # set ds_out.time to mid-interval values
            ds_out = time_set_mid(ds_out, time_name)

            print_timestamp("time_set_mid returned")

            # copy file attributes
            ds_out.attrs = ds0_attrs

            datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            msg = f"{datestamp}: created by {__file__}"
            if "history" in ds_out.attrs:
                ds_out.attrs["history"] = "\n".join([msg, ds_out.attrs["history"]])
            else:
                ds_out.attrs["history"] = msg

            ds_out.attrs["input_file_list"] = " ".join(fnames)

            for key in ["unlimited_dims"]:
                if key in ds0_encoding:
                    ds_out.encoding[key] = ds0_encoding[key]

            # restore encoding for time from first file
            ds_out[time_name].encoding = time_encoding

            # change output units, if specified in var_spec
            units_key = (
                "integral_display_units"
                if tseries_op == "integrate"
                else "display_units"
            )
            if units_key in var_spec:
                ds_out[varname] = conv_units(ds_out[varname], var_spec[units_key])
                print_timestamp("units converted")

            # add regional sum of weights
            ds_out[weight_sum.name] = weight_sum

    print_timestamp("ds_in and client closed")

    # if cluster was instantiated here, close it
    if cluster_in is None:
        cluster.close()

    return ds_out


def test_open_mfdataset(paths, time_chunksize, varname=None):
    for ind in range(len(paths) - 1):
        print(" ".join(["testing open_mfdatset for", paths[ind], paths[ind + 1]]))
        ds = xr.open_mfdataset(
            paths[ind : ind + 2],
            data_vars="minimal",
            combine="by_coords",
            chunks={time_name: time_chunksize},
        )
        if varname is not None:
            print(ds[varname])


def tseries_fname(varname, component, experiment, ensemble, freq):
    """return relative filename for tseries"""
    return f"{varname}_{component}_{experiment}_{ensemble:02d}_{freq}.nc"
