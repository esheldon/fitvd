from __future__ import print_function
from . import files

def load_mask(tilename=None, fname=None):
    assert tilename is not None or fname is not None, \
        'send tilename or fname'

    if fname is None:
        fname = files.get_mask_file(tilename)

    print('loading mask from:', fname)
    return Mask(fname=fname)

class Mask(object):
    def __init__(self, fname):
        self._fname = fname
        self._load_mask()

    def _load_mask(self):
        import healsparse as hs
        self._smap = hs.HealSparseMap.read(self._fname)

    def is_masked(self, ra, dec):
        """
        check if the input positions are masked
        """

        values = self._smap.getValueRaDec(ra, dec)
        return values > 0

    def is_unmasked(self, ra, dec):
        """
        check if the input positions are masked
        """

        is_masked = self.is_masked(ra, dec)
        return ~is_masked


'''
class MaskOld(object):
    def __init__(self, fname, config):
        self.fname = fname
        self._set_config(config)
        self._load_mask()

    def _load_mask(self):
        import smatch
        rad_mult = self.config['rad_mult']

        self.maskdata = read_mask(self.fname)
        if self.maskdata.size > 0:
            self.scat = smatch.Catalog(
                self.maskdata['ra'],
                self.maskdata['dec'],
                rad_mult*self.maskdata['rad_arcsec']/3600.0,
            )
        else:
            self.scat = None

    def is_masked(self, ra, dec):
        """
        check if the input positions are masked
        """
        is_masked = np.zeros(ra.size, dtype='bool')

        if self.maskdata.size == 0:
            return is_masked

        matches = self._match(ra, dec)

        is_masked[matches['i2']] = True
        return is_masked

    def is_unmasked(self, ra, dec):
        """
        check if the input positions are masked
        """

        is_masked = self.is_masked(ra, dec)
        return ~is_masked

    def _match(self, ra, dec):
        """
        match the input positions
        """

        self.scat.match(ra, dec, maxmatch=-1)
        return self.scat.matches

    def _set_config(self, config):
        self.config = {
            'rad_mult': 1.5,
        }
        if config is not None:
            self.config.update(config)


def read_mask(fname):
    """
    load a mask file

    Parameters
    ----------
    fname: string
        path to the file
    """
    print('loading mask from: %s' % fname)

    data = {}
    with open(fname) as fobj:
        for line in fobj:
            line = line.strip()

            if '#' in line or '---' in line:
                continue

            ls = line.split()
            ind = int(ls[0])
            ra = float(ls[1])
            dec = float(ls[2])
            rad_arcsec = float(ls[4])

            if rad_arcsec > 0:

                if ind in data:
                    radmax = max(rad_arcsec, data[ind]['rad_arcsec'])
                    data[ind]['rad_arcsec'] = radmax
                else:
                    data[ind] = {
                        'id': ind,
                        'ra': ra,
                        'dec': dec,
                        'rad_arcsec': rad_arcsec,
                    }
            else:
                print('found zero rad:',line)

    dt = [
        ('id', 'i8'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('rad_arcsec', 'f8'),
    ]

    st = np.zeros(len(data), dtype=dt)

    for i, key in enumerate(data):
        d = data[key]
        st['id'][i] = d['id']
        st['ra'][i] = d['ra']
        st['dec'][i] = d['dec']
        st['rad_arcsec'][i] = d['rad_arcsec']

    return st
'''
