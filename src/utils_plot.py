"""
utility functions related to plotting
"""

import os

import matplotlib.pyplot as plt

from src.utils import is_date, time_year_plus_frac
from src.config import rootdir


def _fig_fname_resolved(fname):
    """
    Given a relative filename fname, return full path for a figure.
    The directory containing the returned path is created, if it doesn't already exist.
    """
    if "TESTMODE" in os.environ and os.environ["TESTMODE"] == "True":
        figdir_relative = "figures_test"
    else:
        figdir_relative = "figures"
    figdir_abs = os.path.join(rootdir, figdir_relative)
    os.makedirs(figdir_abs, exist_ok=True)
    return os.path.join(figdir_abs, fname)


def plot_1var(
    varname,
    ds_list,
    legend_list,
    linestyle_list=None,
    handlelength=None,
    title=None,
    ylabel=True,
    show_legend=True,
    figsize=(10, 6),
    region_val=None,
    vdim_name=None,
    vdim_ind=None,
    fname=None,
    ax=None,
    xoffsets=None,
    yoffsets=None,
    **kwargs,
):
    """
    create a simple plot of a tseries variable for multiple datasets
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    linestyle_ind = 0
    for ds_ind, ds in enumerate(ds_list):
        varname_x = next(dim for dim in ds[varname].dims if dim != "ensemble")
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = "time (years)"
        else:
            xvals = da_x.values
            xlabel = (
                f'{varname_x} ({da_x.attrs["units"]})'
                if "units" in da_x.attrs
                else varname_x
            )
        if xoffsets is not None:
            xvals = xvals.copy() + xoffsets[ds_ind]
        seldict = _seldict(ds, region_val, vdim_name, vdim_ind)

        if "ensemble" in ds[varname].dims:
            for ensemble in range(ds.dims["ensemble"]):
                yvals = ds[varname].sel(seldict).isel(ensemble=ensemble).values
                if yoffsets is not None:
                    yvals = yvals.copy() + yoffsets[ds_ind]
                Line2D_list = ax.plot(
                    xvals,
                    yvals,
                    label=f"{legend_list[ds_ind]}, #{ensemble+1}",
                    **kwargs,
                )
                if linestyle_list is not None:
                    Line2D_list[0].set_linestyle(linestyle_list[linestyle_ind])
                    linestyle_ind = (linestyle_ind + 1) % len(linestyle_list)
        else:
            yvals = ds[varname].sel(seldict).values
            if yoffsets is not None:
                yvals = yvals.copy() + yoffsets[ds_ind]
            Line2D_list = ax.plot(xvals, yvals, label=legend_list[ds_ind], **kwargs)
            if linestyle_list is not None:
                Line2D_list[0].set_linestyle(linestyle_list[linestyle_ind])
                linestyle_ind = (linestyle_ind + 1) % len(linestyle_list)
    ax.set_xlabel(xlabel)
    if ylabel and ds[varname].attrs["units"] != "1":
        ax.set_ylabel(ds[varname].attrs["units"])
    if show_legend:
        ax.legend(handlelength=handlelength)
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(
            _fig_fname_resolved(fname), dpi=600, metadata={"CreationDate": None}
        )
    return ax


def plot_1ds(
    ds,
    varnames,
    title=None,
    figsize=(10, 6),
    region_val=None,
    vdim_name=None,
    vdim_ind=None,
    fname=None,
    ax=None,
):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
    for varname in varnames:
        varname_x = next(dim for dim in ds[varname].dims if dim != "ensemble")
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = "time (years)"
        else:
            xvals = da_x.values
            xlabel = (
                f'{varname_x} ({da_x.attrs["units"]})'
                if "units" in da_x.attrs
                else varname_x
            )
        if "ensemble" in ds[varname].dims:
            for ensemble in range(ds.dims["ensemble"]):
                ax.plot(
                    xvals,
                    ds[varname].sel(seldict).isel(ensemble=ensemble),
                    label=f"{varname}, #{ensemble+1}",
                )
        else:
            ax.plot(xvals, ds[varname].sel(seldict), label=varname)
    ax.set_xlabel(xlabel)
    if ds[varname].attrs["units"] != "1":
        ax.set_ylabel(ds[varname].attrs["units"])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(
            _fig_fname_resolved(fname), dpi=600, metadata={"CreationDate": None}
        )
    return ax


def plot_vars_vs_var(
    ds,
    varname_x,
    varnames_y,
    title=None,
    figsize=(10, 6),
    region_val=None,
    fname=None,
    ax=None,
):
    """
    create a simple plot of a list of tseries variables vs a single tseries variable
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    seldict = _seldict(ds, region_val, vdim_name=None, vdim_ind=None)
    for varname_y in varnames_y:
        if "ensemble" in ds.dims:
            for ensemble in range(ds.dims["ensemble"]):
                ax.plot(
                    ds[varname_x].sel(seldict).isel(ensemble=ensemble),
                    ds[varname_y].sel(seldict).isel(ensemble=ensemble),
                    label=f"{varname_y}, #{ensemble+1}",
                )
        else:
            ax.plot(
                ds[varname_x].sel(seldict), ds[varname_y].sel(seldict), label=varname_y
            )
    ax.set_xlabel(varname_x + "(" + ds[varname_x].attrs["units"] + ")")
    if ds[varname_y].attrs["units"] != "1":
        ax.set_ylabel(ds[varname_y].attrs["units"])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(
            _fig_fname_resolved(fname), dpi=600, metadata={"CreationDate": None}
        )
    return ax


def _seldict(ds, region_val, vdim_name, vdim_ind):
    """
    return dictionary of dimensions and indices
    to be used in sel operator
    """

    seldict = {}
    if region_val is not None:
        seldict["region"] = region_val
    if vdim_name is not None:
        if vdim_ind is None:
            vdim_ind_loc = -1 if vdim_name == "lev" else 0
        else:
            vdim_ind_loc = vdim_ind
        seldict[vdim_name] = ds[vdim_name].values[vdim_ind_loc]

    return seldict
