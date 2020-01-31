"""interface for generating xarray Datasets accesible from data_catalog"""

import xarray as xr
import yaml

from src import data_catalog
from src.utils import print_timestamp, time_set_mid, drop_var_names

from src.config import var_specs_fname

time_name = "time"


def gen_ds_vars(varnames, component, experiment, stream=None, df=None):
    """
    return xarray.Dataset containing varnames
    assumes that data_catalog.set_catalog has been called

    arguments are passed to gen_ds_var
    """
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode="r") as fptr:
            var_specs_all = yaml.safe_load(fptr)
        stream = var_specs_all[component]["stream"]

    # get DataFrame of matching data_catalog entries
    if df is None:
        df = data_catalog.find_in_index(
            variable=varnames, component=component, stream=stream, experiment=experiment
        )

    ds = xr.merge(
        [gen_ds_var(varname, component, experiment, stream, df) for varname in varnames]
    )

    return ds


def gen_ds_var(varname, component, experiment, stream=None, df_in=None):
    """
    return xarray.Dataset containing varname
    assumes that data_catalog.set_catalog has been called
    """
    print_timestamp(f"entering gen_ds_var, varname={varname}, experiment={experiment}")
    # if no stream is specified, get the default stream for this component
    if stream is None:
        with open(var_specs_fname, mode="r") as fptr:
            var_specs_all = yaml.safe_load(fptr)
        stream = var_specs_all[component]["stream"]

    # get DataFrame of matching data_catalog entries
    if df_in is None:
        df = data_catalog.find_in_index(
            variable=varname, component=component, stream=stream, experiment=experiment
        )
    else:
        df = df_in.loc[df_in["variable"] == varname]

    if df.empty:
        raise ValueError(
            f"no file matches found for varname={varname}, component={component}, experiment={experiment}"
        )

    ensembles = df.ensemble.unique()
    if len(ensembles) > 1:
        # concatenate matching ensembles
        ds_list = [
            _gen_ds_var_single_ensemble(
                varname, component, experiment, stream, df, ensemble
            )
            for ensemble in ensembles
        ]
        ds = xr.concat(
            ds_list,
            dim="ensemble",
            data_vars=[varname],
            coords="minimal",
            compat="override",
        )
    else:
        ds = _gen_ds_var_single_ensemble(
            varname, component, experiment, stream, df, ensembles[0]
        )

    return ds


def _gen_ds_var_single_ensemble(
    varname, component, experiment, stream, df_in, ensemble
):
    """
    return xarray.Dataset containing varname for a single ensemble member
    """
    # get DataFrame of matching data_catalog entries
    df = df_in.loc[df_in["ensemble"] == ensemble]

    if df.empty:
        raise ValueError(
            f"no file matches found for varname={varname}, component={component}, experiment={experiment}, ensemble={ensemble}"
        )

    paths = df["files"].tolist()

    with xr.open_dataset(paths[0]) as ds0:
        rank = len(ds0[varname].dims)
        time_chunksize = 10 * 12 if rank < 4 else 12
        time_encoding = ds0[time_name].encoding
        ds_encoding = ds0.encoding
        drop_var_names_loc = drop_var_names(component, ds0, varname)

    ds_out = xr.open_mfdataset(
        paths,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine="by_coords",
        drop_variables=drop_var_names_loc,
    ).chunk(chunks={time_name: time_chunksize})

    for key in ["units", "calendar"]:
        if key in time_encoding:
            ds_out[time_name].encoding[key] = time_encoding[key]

    # set ds_out.time to mid-interval values
    ds_out = time_set_mid(ds_out, time_name)

    for key in ["unlimited_dims"]:
        if key in ds_encoding:
            ds_out.encoding[key] = ds_encoding[key]

    return ds_out
