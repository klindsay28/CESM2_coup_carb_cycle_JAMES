#! /usr/bin/env python3

import os

import pytest
import xarray as xr

import data_catalog
from tseries_mod import tseries_get_vars
from utils_test import dict_skip_keys, ds_identical_skip_attr_list

data_catalog.set_catalog('experiments')

cache_dir_test = 'tseries_test'
os.makedirs(cache_dir_test, exist_ok=True)

@pytest.mark.parametrize('varnames, component, experiment, stream', [
    (['SFCO2_OCN'], 'atm', 'esm-hist-cmip5', 'cam2.h0'),
    (['SFCO2_OCN'], 'atm', 'esm-hist', None),
    (['TOTECOSYSC', 'NBP'], 'lnd', 'esm-hist-cmip5', None),
    (['TOTECOSYSC', 'NBP'], 'lnd', 'esm-hist', None),
    (['FG_CO2', 'ATM_CO2'], 'ocn', 'esm-hist-cmip5', None),
    (['FG_CO2', 'ATM_CO2'], 'ocn', 'esm-hist', None),
])
def test_tseries_get_vars(varnames, component, experiment, stream):
    ds_base = tseries_get_vars(varnames, component, experiment,
                               stream=stream, freq='ann')
    ds_test = tseries_get_vars(varnames, component, experiment,
                               stream=stream, freq='ann',
                               cache_dir=cache_dir_test, clobber=True)
    skip_attr_list = ['history']
    assert ds_identical_skip_attr_list(ds_base, ds_test, skip_attr_list)

    ds_base = tseries_get_vars(varnames, component, experiment,
                               stream=stream, freq='mon')
    ds_test = tseries_get_vars(varnames, component, experiment,
                               stream=stream, freq='mon',
                               cache_dir=cache_dir_test)
    skip_attr_list = ['history']
    assert ds_identical_skip_attr_list(ds_base, ds_test, skip_attr_list)
