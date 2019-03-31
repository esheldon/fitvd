from __future__ import print_function

import numpy as np

class Mask(object):
    def __init__(self, fname, config):
        self.fname=fname
        self._set_config(config)
        self._load_mask()

    def _load_mask(self):
        import smatch
        rad_mult = self.config['rad_mult']

        self.maskdata = read_mask(self.fname)
        self.scat = smatch.Catalog(
            self.maskdata['ra'],
            self.maskdata['dec'],
            rad_mult*self.maskdata['rad_arcsec']/3600.0,
        )

    def is_masked(self, ra, dec):
        """
        check if the input positions are masked
        """

        matches = self._match(ra, dec)

        is_masked = np.zeros(ra.size, dtype='bool')
        is_masked[ matches['i2'] ] = True
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
    tilename: string
        Either the basic tilename such as SN-C3_C10
        or with reqnum/attnum SN-C3_C10_r3688p01
    """
    print('loading mask from: %s' % fname)

    data={}
    with open(fname) as fobj:
        for line in fobj:
            if '#' in line or '---' in line:
                continue

            ls = line.split()
            ind = int( ls[0] )
            ra = float( ls[1] )
            dec = float( ls[2] )
            band = ls[3]
            rad_arcsec = float(ls[4])
            
            if ind in data:
                radmax = max( rad_arcsec, data[ind]['rad_arcsec'])
                data[ind]['rad_arcsec'] = radmax
            else:
                data[ind] = {
                    'id': ind,
                    'ra': ra,
                    'dec': dec,
                    'rad_arcsec': rad_arcsec,
                }


    dt = [
        ('id','i8'),
        ('ra','f8'),
        ('dec','f8'),
        ('rad_arcsec','f8'),
    ]

    st = np.zeros(len(data), dtype=dt)

    for i,key in enumerate(data):
        d = data[key] 
        st['id'][i] = d['id']  
        st['ra'][i] = d['ra']  
        st['dec'][i] = d['dec']  
        st['rad_arcsec'][i] = d['rad_arcsec']  

    return st


