#!/usr/bin/env bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh

# Setup virtual environnment
# Remember to re-activate if you open a new shell
ENVNAME="mg5tut"
python3 -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
export PYTHONPATH="${ENVNAME}/lib/python3.8/site-packages/:${PYTHONPATH}"

