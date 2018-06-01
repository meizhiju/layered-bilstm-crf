#!/usr/bin/env bash

function tuning_hyperparameters()
{
    # Init data for Bayesian Optimization
    minimizer=$1
    acqFunc=$2
    nCalls=$3
    nRandomStarts=$4
    noise=$5
    randomState=$6

    # Tuning arguments for Bayesian Optimization
    acqOpt=$7
    nRestartsOpt=$8
    nPoints=$9
    xi=${10}

    serverNo="${11}"

    # Log file for tracing back later
    output="bayesian_optimization_nrb-gp${serverNo}.out"

    startTime=$(date '+%d/%m/%Y %H:%M:%S')
    echo 'Start time: '$startTime''
    echo 'Dropout Input: '$drInput' --- Output: '$drOutput' --- Adam: '$adam' --- Clipping Threshold: '$threshold' --- Weight Decay: '$decayRate''

    # Using nohup for running this process in the background (if the process takes much time)
    # We can use "tail -f path_to_file" to see the logs
    nohup python3 -u ../src/bayesian_opt.py --minimizer $minimizer --acq-func $acqFunc --n-calls $nCalls --n-random-starts $nRandomStarts --noise $noise --random-state $randomState --acq-opt $acqOpt --n-restarts-opt $nRestartsOpt --n-points $nPoints --xi $xi >> $output &

    # or run it directly
    # python3 ../src/bayesian_opt.py --minimizer $minimizer --acq-func $acqFunc --n-calls $nCalls --n-random-starts $nRandomStarts --noise $noise --random-state $randomState --acq-opt $acqOpt --n-restarts-opt $nRestartsOpt --n-points $nPoints --xi $xi
}

function use_gp_minimize()
{
    server=1            # Narobi server is used for tuning
    minimizer=1            # gp_minimize

    nCalls=10           # Number of calls to our model (set value to 10 in order for demo quickly)
    randomState=-1      # Replace -1 by another number if you want the result is reproducible

    acqFunc="gp_hedge"  # EI, LCB, PI, gp_hedge
    nRandomStarts=10
    noise=-1            # Gaussian noise

    acqOpt="lbfgs"      # "lbfgs" or "sampling"
    nRestartsOpt=100
    nPoints=50000
    xi=0.1

    echo '******************* Server '$server' *******************'
    tuning_hyperparameters $minimizer $acqFunc $nCalls $nRandomStarts $noise $randomState $acqOpt $nRestartsOpt $nPoints $xi $server
}

function use_dummy_minimize()
{
    server=2
    minimizer=2            # dummy_minimize

    nCalls=10
    randomState=-1

    # dummy_minimize doesn't use these arguments but we have to pass them to tuning_hyperparameters function
    acqFunc="0"
    nRandomStarts=0
    noise=0
    acqOpt=""
    nRestartsOpt=0
    nPoints=0
    xi=0

    echo '******************* Server '$server' *******************'
    tuning_hyperparameters $minimizer $acqFunc $nCalls $nRandomStarts $noise $randomState $acqOpt $nRestartsOpt $nPoints $xi $server
}

function use_forest_minimize()
{
    server=3
    minimizer=3            # forest_minimize

    nCalls=10
    randomState=-1

    acqFunc="LCB"        # EI, LCB, PI
    nRandomStarts=10

    nPoints=20000
    xi=0.1

    # forest_minimize doesn't use this argument but we have to pass them to tuning_hyperparameters function
    noise=0
    acqOpt=""
    nRestartsOpt=0

    echo '******************* Server '$server' *******************'
    tuning_hyperparameters $minimizer $acqFunc $nCalls $nRandomStarts $noise $randomState $acqOpt $nRestartsOpt $nPoints $xi $server
}

function use_gbrt_minimize()
{
    server=4
    minimizer=4            # gbrt_minimize

    nCalls=10
    randomState=-1

    acqFunc="LCB"        # EI, LCB, PI
    nRandomStarts=10

    nPoints=20000
    xi=0.1

    # forest_minimize doesn't use this argument but we have to pass them to tuning_hyperparameters function
    noise=0
    acqOpt=""
    nRestartsOpt=0

    echo '******************* Server '$server' *******************'
    tuning_hyperparameters $minimizer $acqFunc $nCalls $nRandomStarts $noise $randomState $acqOpt $nRestartsOpt $nPoints $xi $server
}

dateTime=$(date '+%d/%m/%Y %H:%M:%S')
echo '********************************************************'
echo '***** START TUNING PROCESS AT '$dateTime' *****'
echo '********************************************************'

# Call 1 in 4 function for tuning hyperparameters
use_gp_minimize
#use_dummy_minimize
#use_forest_minimize
#use_gbrt_minimize

# ****************************************************************************************

