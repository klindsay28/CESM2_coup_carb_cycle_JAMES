"""settings used in multiple .py and .ipynb files"""

import os

srcdir = os.path.abspath(os.path.dirname(__file__))

rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))

forcing_dir = os.path.join(rootdir, "forcing")

grid_dir = os.path.join(rootdir, "grid")

obspack_dir = os.path.join(
    os.path.sep,
    "glade",
    "work",
    "klindsay",
    "analysis",
    "obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12",
)

var_specs_fname = os.path.join(rootdir, "var_specs.yaml")

expr_metadata_fname = os.path.join(rootdir, "expr_metadata.yaml")
