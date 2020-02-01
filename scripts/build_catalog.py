#! /usr/bin/env python

from src.data_catalog import build_catalog
from src.config import expr_metadata_fname

build_catalog(expr_metadata_fname, clobber=True)
