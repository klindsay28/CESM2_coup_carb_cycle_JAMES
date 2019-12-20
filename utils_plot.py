"""
utility functions related to plotting
"""

import matplotlib.pyplot as plt

from utils import is_date, time_year_plus_frac

def plot_1var(varname, ds_list, legend_list, title=None, figsize=(10,6), region_val=None, vdim_name=None, vdim_ind=None, fname=None):
    """
    create a simple plot of a tseries variable for multiple datasets
    use units from last tseries variable for ylabel
    """
    fig, ax = plt.subplots(figsize=figsize)
    for ds_ind, ds in enumerate(ds_list):
        varname_x = ds[varname].dims[0]
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = 'time (years)'
        else:
            xvals = da_x.values
            xlabel = f'{varname_x} ({da_x.attrs["units"]})' if 'units' in da_x.attrs else varname_x
        seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
        ax.plot(xvals, ds[varname].sel(seldict), label=legend_list[ds_ind])
    ax.set_xlabel(xlabel)
    if ds[varname].attrs['units'] != '1':
        ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600)

def plot_1ds(ds, varnames, title=None, figsize=(10,6), region_val=None, vdim_name=None, vdim_ind=None, fname=None):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    fig, ax = plt.subplots(figsize=figsize)
    seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
    for varname in varnames:
        varname_x = ds[varname].dims[0]
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = 'time (years)'
        else:
            xvals = da_x.values
            xlabel = f'{varname_x} ({da_x.attrs["units"]})' if 'units' in da_x.attrs else varname_x
        if region_val is None:
            ax.plot(xvals, ds[varname], label=varname)
        else:
            ax.plot(xvals, ds[varname].sel(seldict), label=varname)
    ax.set_xlabel(xlabel)
    if ds[varname].attrs['units'] != '1':
        ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600)

def plot_vars_vs_var(ds, varname_x, varnames_y, title=None, figsize=(10,6), region_val=None, fname=None):
    """
    create a simple plot of a list of tseries variables vs a single tseries variable
    use units from last tseries variable for ylabel
    """
    fig, ax = plt.subplots(figsize=figsize)
    for varname_y in varnames_y:
        if region_val is None:
            ax.plot(ds[varname_x], ds[varname_y], label=varname_y)
        else:
            ax.plot(ds[varname_x].sel(region=region_val), ds[varname_y].sel(region=region_val), label=varname_y)
    ax.set_xlabel(varname_x + '(' + ds[varname_x].attrs['units'] + ')')
    if ds[varname_y].attrs['units'] != '1':
        ax.set_ylabel(ds[varname_y].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600)

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
