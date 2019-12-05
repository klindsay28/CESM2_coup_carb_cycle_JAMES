"""
values from CIME's shr_const_mod.F90

Fortran source code available at
https://github.com/ESMCI/cime/blob/master/src/share/util/shr_const_mod.F90
"""

import xarray as xr

_CIME_shr_const_dict = {
    'pi': xr.DataArray(3.14159265358979323846,
                      attrs={'long_name': 'pi',
                             'units': '1'}),
    'g': xr.DataArray(9.80616,
                      attrs={'long_name': "acceleration from Earth's gravity",
                             'units': 'm/s2'}),
    'rearth': xr.DataArray(6.37122e6,
                           attrs={'long_name': "Earth's radius",
                                  'units': 'm'}),
    'mwdair': xr.DataArray(28.966,
                           attrs={'long_name': 'molecular weight of dry air',
                                  'units': 'g/mol'}),    
}

def CIME_shr_const(name):
    """return a value from CIME's shr_const_mod.F90"""
    return _CIME_shr_const_dict[name]
