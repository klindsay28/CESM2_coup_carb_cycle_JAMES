#! /usr/bin/env python

from data_catalog import build_catalog
from config import expr_metadata_fname

build_catalog(expr_metadata_fname, clobber=True)
