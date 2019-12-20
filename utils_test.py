#! /usr/bin/env python3

import esmlab_wrap

import utils
import utils_units

print(utils_units.clean_units('years'))
print(utils_units.clean_units('degC'))
print(utils_units.clean_units('gC/gN'))
print(utils_units.clean_units('meq/m^3'))

for decode_times in [False, True]:
    print('******************************')
    print('decode_times = %s' % decode_times)
    ds = utils.xr_ds_ex(decode_times)

    ds_ann = esmlab_wrap.compute_ann_mean(ds)

    print(ds)
    print('**********')
    print('ds.time time_year_plus_frac')
    print(utils.time_year_plus_frac(ds, 'time')[0:12])
    print('********************')
    print(ds_ann)
    print('**********')
    print('ds_ann.time time_year_plus_frac')
    print(utils.time_year_plus_frac(ds_ann, 'time'))
