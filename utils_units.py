"""
arithmetic of xarray DataArray's with units

This is ad-hoc.
"""

import cf_units
import xarray as xr

def mult_w_units(var_a, var_b):
    """
    multiply var_a and var_b, combining units
    """
    da_out = var_a * var_b
    if not isinstance(da_out, xr.DataArray):
        raise ValueError('product of var_a and var_b must be an xr.DataArray')
    a_units = _get_units(var_a)
    b_units = _get_units(var_b)
    da_out.attrs['units'] = cf_units.Unit(f'({a_units})*({b_units})').format()
    return da_out

def div_w_units(var_a, var_b):
    """
    divide var_a and var_b, combining units
    """
    da_out = var_a / var_b
    if not isinstance(da_out, xr.DataArray):
        raise ValueError('quotient of var_a and var_b must be an xr.DataArray')
    a_units = _get_units(var_a)
    b_units = _get_units(var_b)
    da_out.attrs['units'] = cf_units.Unit(f'({a_units})/({b_units})').format()
    return da_out

def add_w_units(var_a, var_b):
    """
    add var_a and var_b, verifying that units are compatible
    """
    da_out = var_a + var_b
    if not isinstance(da_out, xr.DataArray):
        raise ValueError('sum of var_a and var_b must be an xr.DataArray')
    a_units = _get_units(var_a)
    b_units = _get_units(var_b)
    # verify that normalized units are the same
    if cf_units.Unit(a_units).format() != cf_units.Unit(b_units).format():
        msg = f'incompatible units, {a_units} != {b_units}'
        raise ValueError(msg)
    da_out.attrs['units'] = a_units
    return da_out

def subtract_w_units(var_a, var_b):
    """
    subtract var_a and var_b, verifying that units are compatible
    """
    da_out = var_a - var_b
    if not isinstance(da_out, xr.DataArray):
        raise ValueError('difference of var_a and var_b must be an xr.DataArray')
    a_units = _get_units(var_a)
    b_units = _get_units(var_b)
    # verify that normalized units are the same
    if cf_units.Unit(a_units).format() != cf_units.Unit(b_units).format():
        msg = f'incompatible units, {a_units} != {b_units}'
        raise ValueError(msg)
    da_out.attrs['units'] = a_units
    return da_out

def _get_units(var):
    """
    return units of var
    return '1' if var is a float
    """
    return '1' if isinstance(var, float) else var.attrs['units']
