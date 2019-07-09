import sys
import logging
import numpy as np
import ngmix
from . import procflags

logger = logging.getLogger(__name__)


def setup_logging(level):
    if level == 'info':
        level = logging.INFO
    elif level == 'debug':
        level = logging.DEBUG
    elif level == 'warning':
        level = logging.WARNING
    elif level == 'error':
        level = logging.ERROR
    else:
        level = logging.CRITICAL

    logging.basicConfig(stream=sys.stdout, level=level)


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
        if front == '':
            front = None
        if back == '' or back == 'noshear':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

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

    ntrials_per = int(round(tmsec/sec_per))

    nsplit = int(ceil(ntrials/float(ntrials_per)))

    time_hours = ntrials_per*sec_per/3600.0

    logger.info("ntrials requested: %s" % (ntrials))
    logger.info('seconds per image: %s sec per with rand: %s' %
                (c['sec_per'], sec_per))
    logger.info('nsplit: %d ntrials per: %d time (hours): %s' %
                (nsplit, ntrials_per, time_hours))

    return ntrials_per, nsplit, time_hours


def get_trials_per_job_mpi(njobs, ntrials):
    """
    split for mpi
    """
    return int(round(float(ntrials)/njobs))


def zero_bitmask_in_weight(mbobs, flags2zero):
    """
    check if the input flags are set in the bmask, if
    so zero the weight map
    """

    new_mbobs = ngmix.MultiBandObsList()
    new_mbobs.meta.update(mbobs.meta)

    for band, obslist in enumerate(mbobs):

        new_obslist = ngmix.ObsList()
        new_obslist.meta.update(obslist.meta)

        for epoch, obs in enumerate(obslist):
            try:
                if obs.has_bmask():
                    bmask = obs.bmask
                    w = np.where((bmask & flags2zero) != 0)
                    if w[0].size > 0:
                        weight = obs.weight
                        logging.debug('band %d epoch %d zeroing %d/%d in '
                                      'weight' %
                                      (band, epoch, w[0].size, bmask.size))
                        weight[w] = 0.0

                        # trigger rebuild of pixels
                        obs.weight = weight
                new_obslist.append(obs)
            except ngmix.GMixFatalError:
                logging.info('band %d epoch %d all zero weight after '
                             'bitmask' % (band, epoch))

        if len(new_obslist) == 0:
            return None, procflags.HIGH_MASKFRAC

        new_mbobs.append(new_obslist)

    return new_mbobs, 0


def get_masked_frac_sums(obs):
    weight = obs.weight
    wmasked = np.where(weight <= 0.0)
    nmasked = wmasked[0].size
    npix = weight.size

    return npix, nmasked


def get_masked_frac(mbobs):
    nmasked = 0.0
    npix = 0

    for obslist in mbobs:
        for obs in obslist:
            tnpix, tnmasked = get_masked_frac_sums(obs)
            nmasked += tnmasked
            npix += tnpix

    masked_frac = nmasked/npix
    return masked_frac


def convert_string_to_seed(string):
    """
    convert the input string to an integer for use
    as a seed
    """
    import hashlib

    h = hashlib.sha256(string.encode('utf-8')).hexdigest()
    seed = int(h, base=16) % 2**30

    logger.info("got seed %d from string %s" % (seed, string))

    return seed


def check_blacklist(mbobs, blacklist):
    """
    check the meta['file_path'] entry against the blacklist

    return a new mbobs without the blacklisted observations
    """

    new_mbobs = ngmix.MultiBandObsList()
    new_mbobs.meta.update(mbobs.meta)

    for band, obslist in enumerate(mbobs):

        new_obslist = ngmix.ObsList()
        new_obslist.meta.update(obslist.meta)

        for epoch, obs in enumerate(obslist):
            file_path = obs.meta['file_path']

            if file_path in blacklist:
                logger.debug('removing blacklisted obs from "%s"' %
                             file_path)
            else:
                new_obslist.append(obs)

        if len(new_obslist) == 0:
            logger.debug('all epochs from band %s are blacklisted' % band)
            return None

        new_mbobs.append(new_obslist)

    return new_mbobs
