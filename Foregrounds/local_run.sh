#!/bin/bash

# A flexible script to run a specific part of the pipeline locally.
# It sets the ${EXPERIMENT_ROOT} env variable which is used in the config file.
# This is helpful if this pipeline is run on an HPC cluster, and the ${EXPERIMENT_ROOT} 
# can be set in terms of ${SCRATCH}. 
#
# Usage:
# ./local_run.sh <run_code> <script_name>
#
# Examples:
# ./local_run.sh COSMIC gen_catalog.py
# ./local_run.sh COSMIC main_loop.py

set -e

# --- 1. VALIDATE INPUT ARGUMENTS ---
if [ "$#" -ne 2 ]; then
    echo "ERROR: Incorrect number of arguments."
    echo "Usage: $0 <run_code> <script_name>"
    echo "Available scripts: gen_catalog.py, gen_waveforms.py, main_loop.py"
    exit 1
fi

RUN_CODE=$1
SCRIPT_TO_RUN=$2

# --- 2. SETUP: Define the Environment and Prepare the Output Directory ---
export EXPERIMENT_ROOT="${SCRATCH}/projects/ucb-catalogs/confusion_test"
#export EXPERIMENT_ROOT="./"

echo "========================================================="
echo "Starting local run: ${RUN_CODE}"
echo "Executing script: ${SCRIPT_TO_RUN}"
echo "Output folders will be in: ${EXPERIMENT_ROOT}"

# Create the top-level directory. The -p flag means it does nothing if it already exists.
# This is safe for re-running steps.
echo "${EXPERIMENT_ROOT}"
mkdir -p "${EXPERIMENT_ROOT}"
echo "========================================================="
echo

# --- 3. EXECUTION: Run the selected Python script ---
case "${SCRIPT_TO_RUN}" in
    gen_catalog.py)
        echo "--- Running Catalog Generation ---"
        python gen_catalog.py --code "${RUN_CODE}"
        ;;

    gen_waveforms.py)
        echo "--- Running Waveform Generation ---"
        python gen_waveforms.py --code "${RUN_CODE}"
        ;;

    main_loop.py)
        echo "--- Running Main Loop ---"
        python main_loop.py --code "${RUN_CODE}"
        ;;

    *)
        # This is the default case if no other pattern matches.
        echo "Error: Unknown script name '${SCRIPT_TO_RUN}'"
        echo "Available scripts: gen_catalog.py, gen_waveforms.py, main_loop.py"
        exit 1
        ;;
esac

echo
echo "========================================================="
echo "Script '${SCRIPT_TO_RUN}' finished successfully."
echo "========================================================="
