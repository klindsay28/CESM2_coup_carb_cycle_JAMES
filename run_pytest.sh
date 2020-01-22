#!/bin/bash -i

conda activate CESM2_coup_carb_cycle_JAMES

cmdname=$0

# parse command line arguments
pytest_opts=-rs
while [ $# -gt 0 ] ; do
    case "$1" in
        --run_campaign_required)
            if [ -d /glade/campaign ]; then
                pytest_opts="$pytest_opts --run_campaign_required"
            else
                echo campaign not available, --run_campaign_required ignored
            fi
            ;;
        
        *)
            echo $cmdname: unknown option $1
            exit 1
            ;;
    esac
    shift
done

echo pytest_opts=$pytest_opts

pytest $pytest_opts tests
