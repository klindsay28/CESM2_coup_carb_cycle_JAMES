
begin

case = "b40.prescribed_carb.001"
fix_fnames_tails := \
    (/ ".000101-000912.nc", ".001001-001912.nc", ".002001-002912.nc", \
       ".003001-003912.nc", ".004001-004912.nc", ".005001-005912.nc", \
       ".006001-006912.nc", ".007001-007912.nc", ".008001-008912.nc", \
       ".009001-009912.nc", ".010001-010912.nc" /)

case = "b40.coup_carb.004"
fix_fnames_tails := \
    (/ ".000101-000912.nc", ".001001-001912.nc", ".002001-002912.nc", \
       ".003001-003912.nc", ".004001-004912.nc", ".005001-005912.nc", \
       ".006001-006912.nc", ".007001-007912.nc", ".008001-008912.nc", \
       ".009001-009912.nc", ".010001-010912.nc", ".011001-011912.nc", \
       ".012001-012912.nc", ".013001-013912.nc", ".014001-014912.nc", \
       ".015001-015912.nc", ".016001-016912.nc", ".017001-017912.nc", \
       ".018001-018912.nc", ".019001-019912.nc", ".020001-020912.nc", \
       ".021001-021912.nc", ".022001-022912.nc", ".023001-023912.nc", \
       ".024001-024912.nc", ".025001-025912.nc", ".026001-026912.nc", \
       ".027001-027912.nc", ".028001-028912.nc", ".029001-029912.nc", \
       ".030001-030912.nc" /)

file_varnames = (/ "DIC", "DOC", "TEMP", "POC_FLUX_IN", "POC_FLUX_100m", \
                   "photoC_diat", "photoC_sp", "photoC_diaz", \
                   "photoC_diat_zint_100m", "photoC_sp_zint_100m", "photoC_diaz_zint_100m" /)

fix_varnames = (/ "moc_components", "transport_components", "transport_regions" /)

do file_varname_ind = 0, dimsizes(file_varnames)-1
    file_varname = file_varnames(file_varname_ind)

    fname_in = case + ".pop.h." + file_varname + ".100001-100012.nc"
    fnames = case + ".pop.h." + file_varname + fix_fnames_tails

    ds_in = addfile(fname_in, "r")
    do fname_ind = 0, dimsizes(fnames)-1
        fname = fnames(fname_ind)
        print(fname)
        ds = addfile(fname, "w")
        do fix_varname_ind = 0, dimsizes(fix_varnames)-1
            fix_varname = fix_varnames(fix_varname_ind)
            ds->$fix_varname$ = ds_in->$fix_varname$
        end do
    end do
end do

end
