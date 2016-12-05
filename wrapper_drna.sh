#!/bin/bash

params="$*"

exec python3 drna.py ${params} > stdout.log 2> stderr.log
