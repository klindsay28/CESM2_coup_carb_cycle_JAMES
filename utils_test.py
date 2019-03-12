#! /usr/bin/env python3

import esmlab

import utils

print(utils.clean_units('years'))
print(utils.clean_units('degC'))
print(utils.clean_units('gC/gN'))
print(utils.clean_units('meq/m^3'))

for encode_time in [False, True]:
    print('encode_time = %s' % encode_time)
    ds = utils.xr_ds_ex(encode_time)
    print('ds.time vals')
    print(ds.time[0:12])
    print('ds.time time_year_plus_frac')
    print(utils.time_year_plus_frac(ds, 'time')[0:12])

    ds_ann = esmlab.climatology.compute_ann_mean(ds)
    print('ds_ann.time vals')
    print(ds_ann.time)
    print('ds_ann.time time_year_plus_frac')
    print(utils.time_year_plus_frac(ds_ann, 'time'))
