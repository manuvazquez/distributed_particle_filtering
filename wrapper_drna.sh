#!/bin/bash

unset LD_LIBRARY_PATH

params="$*"

exec python3 DRNA/drna.py ${params} > stdout.log 2> stderr.log
