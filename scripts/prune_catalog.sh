#!/bin/sh

# attempt to cd to directory with catalog
cd `git rev-parse --show-toplevel`/lib_data_catalog

if ! [ -f cesm_coupled.csv.gz ]; then
    echo "did not succeed in cd'ing to directory with catalog"
    exit 1
fi

CASE=b.e21.B1850.f09_g17.CMIP6-piControl.001
# annual ptrns to remove
ptrn1="$CASE.*\.1[5-9][0-9][0-9]-[0-9][0-9][0-9][0-9].nc"
ptrn2="$CASE.*\.[2-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9].nc"
# monthly ptrns to remove
ptrn3="$CASE.*\.1[5-9][0-9][0-9][01][0-9]-[0-9][0-9][0-9][0-9][01][0-9].nc"
ptrn4="$CASE.*\.[2-9][0-9][0-9][0-9][01][0-9]-[0-9][0-9][0-9][0-9][01][0-9].nc"
# daily ptrns to remove
ptrn5="$CASE.*\.1[5-9][0-9][0-9][01][0-9][0123][0-9]-[0-9][0-9][0-9][0-9][01][0-9][0123][0-9].nc"
ptrn6="$CASE.*\.[2-9][0-9][0-9][0-9][01][0-9][0123][0-9]-[0-9][0-9][0-9][0-9][01][0-9][0123][0-9].nc"
zcat cesm_coupled.csv.gz | grep -vE "$ptrn1|$ptrn2|$ptrn3|$ptrn4|$ptrn5|$ptrn6" > cesm_coupled.csv
zcat cesm_coupled.csv.gz | grep -E  "$ptrn1|$ptrn2|$ptrn3|$ptrn4|$ptrn5|$ptrn6" > cesm_coupled.deleted_entries.csv
mv cesm_coupled.csv.gz cesm_coupled.csv.bak.gz
gzip cesm_coupled.deleted_entries.csv cesm_coupled.csv
