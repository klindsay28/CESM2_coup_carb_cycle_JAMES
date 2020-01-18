"""CAM specific utility functions"""

import cf_units
import xarray as xr

from CIME_shr_const import CIME_shr_const
from utils_units import conv_units, mult_w_units, div_w_units, subtract_w_units

def CAM_dry_mass_model_avg(ds):
    """dry mass of model domain for CAM"""

    # pressure at top of model, depends on model vertical coordinate
    tom = mult_w_units(ds['hyai'].values[0], ds['P0'])

    # average dry mass of atmosphere, includes mass above model top
    # hard-coded in CAM
    drym = xr.DataArray(98288.0, attrs={'units': 'Pa'})

    mass_model = div_w_units(subtract_w_units(drym, tom), CIME_shr_const('g'))
    mass_model.attrs['long_name'] = 'atmosphere dry mass, over model domain'

    return mass_model

def CAM_kg_to_dry_vmr(ds, mw):
    """compute conversion factor going from kg to dry vmr"""

    surf_area = ds['weight_sum']

    dry_mass_model = mult_w_units(CAM_dry_mass_model_avg(ds), surf_area)
    moles_dair = conv_units(div_w_units(dry_mass_model, CIME_shr_const('mwdair')), 'mol')

    g_per_kg = xr.DataArray(1.0e3, attrs={'units': 'g/kg'})

    conv_factor = div_w_units(g_per_kg, mult_w_units(mw, moles_dair))
    return conv_factor
