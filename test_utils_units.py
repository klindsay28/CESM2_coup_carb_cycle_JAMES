#! /usr/bin/env python3

import utils_units

# basic example, straight from dictionary
print(utils_units.clean_units('years') == 'common_years')

# ensure 'gC' in 'degC' doesn't get converted
print(utils_units.clean_units('degC') == 'degC')

# matches within expressions
print(utils_units.clean_units('gC/gN') == 'g/g')
print(utils_units.clean_units('meq/m^3') == 'mmol/m^3')
