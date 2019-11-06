#!/bin/tcsh

setenv CLOBBER $1
echo CLOBBER=$CLOBBER

conda activate CESM2_coup_carb_cycle_JAMES_tst

set nb_fnames = `grep -l "'ocn'" *.ipynb | grep -v Untitled`
foreach nb_fname ( $nb_fnames )
    echo executing $nb_fname
    jupyter nbconvert --to notebook --inplace \
        --ExecutePreprocessor.kernel_name=python \
        --ExecutePreprocessor.timeout=3600 \
        --execute $nb_fname
end
