"""
utility functions related to plotting
"""

import matplotlib.pyplot as plt

from utils import is_date, time_year_plus_frac

def plot_1var(varname, ds_list, legend_list, title=None, figsize=(10,6), region_val=None, vdim_name=None, vdim_ind=None, fname=None, ax=None, xoffsets=None, yoffsets=None):
    """
    create a simple plot of a tseries variable for multiple datasets
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for ds_ind, ds in enumerate(ds_list):
        varname_x = next(dim for dim in ds[varname].dims if dim!='ensemble')
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = 'time (years)'
        else:
            xvals = da_x.values
            xlabel = f'{varname_x} ({da_x.attrs["units"]})' if 'units' in da_x.attrs else varname_x
        if xoffsets is not None:
            xvals = xvals.copy() + xoffsets[ds_ind]
        seldict = _seldict(ds, region_val, vdim_name, vdim_ind)

        if 'ensemble' in ds[varname].dims:
            for ensemble in range(ds.dims['ensemble']):
                yvals = ds[varname].sel(seldict).isel(ensemble=ensemble).values
                if yoffsets is not None:
                    yvals = yvals.copy() + yoffsets[ds_ind]
                ax.plot(xvals, yvals, label=f'{legend_list[ds_ind]}, #{ensemble+1}')
        else:
            yvals = ds[varname].sel(seldict).values
            if yoffsets is not None:
                yvals = yvals.copy() + yoffsets[ds_ind]
            ax.plot(xvals, yvals, label=legend_list[ds_ind])
    ax.set_xlabel(xlabel)
    if ds[varname].attrs['units'] != '1':
        ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600, metadata={'CreationDate': None})
    return ax

def plot_1ds(ds, varnames, title=None, figsize=(10,6), region_val=None, vdim_name=None, vdim_ind=None, fname=None, ax=None):
    """
    create a simple plot of a list of tseries variables
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    seldict = _seldict(ds, region_val, vdim_name, vdim_ind)
    for varname in varnames:
        varname_x = next(dim for dim in ds[varname].dims if dim!='ensemble')
        da_x = ds[varname_x]
        if is_date(da_x):
            xvals = time_year_plus_frac(ds, varname_x)
            xlabel = 'time (years)'
        else:
            xvals = da_x.values
            xlabel = f'{varname_x} ({da_x.attrs["units"]})' if 'units' in da_x.attrs else varname_x
        if 'ensemble' in ds[varname].dims:
            for ensemble in range(ds.dims['ensemble']):
                ax.plot(xvals, ds[varname].sel(seldict).isel(ensemble=ensemble), label=f'{varname}, #{ensemble+1}')
        else:
            ax.plot(xvals, ds[varname].sel(seldict), label=varname)
    ax.set_xlabel(xlabel)
    if ds[varname].attrs['units'] != '1':
        ax.set_ylabel(ds[varname].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600, metadata={'CreationDate': None})
    return ax

def plot_vars_vs_var(ds, varname_x, varnames_y, title=None, figsize=(10,6), region_val=None, fname=None, ax=None):
    """
    create a simple plot of a list of tseries variables vs a single tseries variable
    use units from last tseries variable for ylabel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    seldict = _seldict(ds, region_val, vdim_name=None, vdim_ind=None)
    for varname_y in varnames_y:
        if 'ensemble' in ds.dims:
            for ensemble in range(ds.dims['ensemble']):
                ax.plot(ds[varname_x].sel(seldict).isel(ensemble=ensemble), ds[varname_y].sel(seldict).isel(ensemble=ensemble),
                        label=f'{varname_y}, #{ensemble+1}')
        else:
            ax.plot(ds[varname_x].sel(seldict), ds[varname_y].sel(seldict), label=varname_y)
    ax.set_xlabel(varname_x + '(' + ds[varname_x].attrs['units'] + ')')
    if ds[varname_y].attrs['units'] != '1':
        ax.set_ylabel(ds[varname_y].attrs['units'])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if fname is not None:
        plt.savefig(fname, dpi=600, metadata={'CreationDate': None})
    return ax

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
