#!/bin/sh

# This script will setup the lsst stack before executing the command
source /opt/lsst/software/stack/loadLSST.bash
setup lsst_distrib

# Hand off to the CMD
exec "$@"
