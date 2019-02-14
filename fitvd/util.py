import sys
import logging
import numpy as np
import ngmix

logger = logging.getLogger(__name__)

def setup_logging(level):
    if level=='info':
        l=logging.INFO
    elif level=='debug':
        l=logging.DEBUG
    elif level=='warning':
        l=logging.WARNING
    elif level=='error':
        l=logging.ERROR
    else:
        l=logging.CRITICAL

    logging.basicConfig(stream=sys.stdout, level=l)

class NoDataError(Exception):
    """
    there was no data
    """
    def __init__(self, value):
        super(NoDataError, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front=='':
            front=None
        if back=='' or back=='noshear':
            back=None

        self.front=front
        self.back=back

        if self.front is None and self.back is None:
            self.nomod=True
        else:
            self.nomod=False



    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)
        
        return n


def get_trials_nsplit(c):
    """
    split into chunks
    """
    from math import ceil

    ntrials = c['ntrials']

    tmsec = c['desired_hours']*3600.0

    sec_per = c['sec_per']

    ntrials_per = int(round( tmsec/sec_per ) )

    nsplit = int(ceil( ntrials/float(ntrials_per) ))

    time_hours = ntrials_per*sec_per/3600.0

    logger.info("ntrials requested: %s" % (ntrials))
    logger.info('seconds per image: %s sec per with rand: %s' % (c['sec_per'],sec_per))
    logger.info('nsplit: %d ntrials per: %d time (hours): %s' % (nsplit,ntrials_per,time_hours))


    return ntrials_per, nsplit, time_hours

def get_trials_per_job_mpi(njobs, ntrials):
    """
    split for mpi
    """
    return int(round(float(ntrials)/njobs))

def get_masked_frac_sums(obs):
    weight = obs.weight
    wmasked = np.where(weight <= 0.0)
    nmasked = wmasked[0].size
    npix = weight.size

    return npix, nmasked

def get_masked_frac(mbobs):
    nmasked=0.0
    npix=0

    for obslist in mbobs:
        for obs in obslist:
            tnpix, tnmasked = get_masked_frac_sums(obs)
            nmasked += tnmasked
            npix += tnpix

    masked_frac = nmasked/npix
    return masked_frac


