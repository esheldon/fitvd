"""
Original code by Matt Becker, modified to use ra/dec and radius
by Erin Sheldon
"""
from __future__ import print_function

import os
import copy
import numpy as np
import esutil as eu

from .pbar import prange


def get_fofs(cat, fof_conf, mask=None):
    """
    generate FoF groups

    Parameters
    ----------
    cat: array with fields
        Usually the cat from a meds file
    fof_conf: dict
        configuration for the FoF group finder
    mask: mask object
        e.g. a star mask.  Objects that are masked are put into
        their own FoF group, and are not allowed to be part
        of a group with other objects
    """

    if mask is not None:
        is_masked = mask.is_masked(cat['ra'], cat['dec'])
    else:
        is_masked = None

    mn = MEDSNbrs(
        cat,
        fof_conf,
        is_masked=is_masked,
    )

    nbr_data = mn.get_nbrs()

    nf = NbrsFoF(nbr_data)
    fofs = nf.get_fofs()

    add_dt = [('mask_flags', 'i4')]
    fofs = eu.numpy_util.add_fields(fofs, add_dt)

    if mask is not None:
        mcat, mfofs = eu.numpy_util.match(cat['number'], fofs['number'])
        assert mcat.size == cat.size
        fofs['mask_flags'][mfofs] = mask.get_mask_flags(
            cat['ra'][mcat],
            cat['dec'][mcat],
        )

    return nbr_data, fofs


def make_singleton_fofs(cat):
    """
    generate a fofs file, one object per groups

    Parameters
    ----------
    cat: array with fields
        Should have a 'number' entry

    Returns
    -------
    Fof group array with fields, entries 'fofid', 'number'

    """
    dt = [('fofid', 'i8'), ('number', 'i8')]
    fofs = np.zeros(cat.size, dtype=dt)
    fofs['fofid'] = np.arange(fofs.size)
    fofs['number'] = cat['number']
    return fofs


class MEDSNbrs(object):
    """
    Gets nbrs of any postage stamp in the MEDS.

    A nbr is defined as any stamp which overlaps the stamp under consideration
    given a buffer or is in the seg map. See the code below.

    Options:
        buff_type - how to compute buffer length for stamp overlap
            'min': minimum of two stamps
            'max': max of two stamps
            'tot': sum of two stamps

        buff_frac - fraction by whch to multiply the buffer

        maxsize_to_replace - postage stamp size to replace with maxsize
        maxsize - size ot use instead of maxsize_to_replace to compute overlap

        check_seg - use object's seg map to get nbrs in addition to postage stamp overlap
    """

    def __init__(self, meds, conf, is_masked=None):
        self.meds = meds
        self.conf = conf

        if is_masked is None:
            is_masked = np.zeros(meds.size, dtype=bool)
        self.is_masked = is_masked
        self.is_unmasked = ~is_masked

        self._init_bounds()

    def _init_bounds(self):
        if self.conf['method'] == 'radius':
            return self._init_bounds_by_radius()
        else:
            raise NotImplementedError('stamps not implemented for ra,dec version')
            return self._init_bounds_by_stamps()

    def _init_bounds_by_radius(self):


        # might be shifted
        ra, dec = self._get_shifted_positions()
        r = self._get_radius()

        med_ra = np.median(ra)
        med_dec = np.median(dec)
        ra_diff = (ra - med_ra) * 3600.0
        dec_diff = (dec - med_dec) * 3600.0

        cosdec = np.cos( np.radians(dec) )
        self.l = (ra_diff - r)*cosdec
        self.r = (ra_diff + r)*cosdec
        self.b = (dec_diff - r)
        self.t = (dec_diff + r)

    def _get_radius(self):
        """
        get radius for offset calculations
        """

        radius_name=self.conf['radius_column']

        min_radius=self.conf.get('min_radius_arcsec',None)
        if min_radius is None:
            # arcsec
            min_radius=1.0

        max_radius=self.conf.get('max_radius_arcsec',None)
        if max_radius is None:
            max_radius=np.inf

        m=self.meds

        r = m[radius_name].copy()

        r *= self.conf['radius_mult']

        r.clip(min=min_radius, max=max_radius, out=r)

        r += self.conf['padding_arcsec']

        return r

    def _get_shifted_positions(self):
        """
        to simplify the FoF calculations we want to be away
        from ra/dec boundaries. we assume the ra range covered
        by this data is not larger than 20 degrees

        We also assume we are not close to the poles, so dec is not modified.
        """
        m=self.meds

        ra = m['ra'].copy()
        dec = m['dec'].copy()

        min_ra = ra.min()
        max_ra = ra.max()

        if min_ra < 20 or max_ra > 340:
            ra = eu.coords.shiftra(ra, shift=180)

        return ra, dec


    def get_nbrs(self,verbose=True):
        nbrs_data = []
        dtype = [('number','i8'),('nbr_number','i8')]

        for mindex in prange(self.meds.size):
            nbrs = self.check_mindex(mindex)

            #add to final list
            for nbr in nbrs:
                nbrs_data.append((self.meds['number'][mindex],nbr))

        #return array sorted by number
        nbrs_data = np.array(nbrs_data,dtype=dtype)
        i = np.argsort(nbrs_data['number'])
        nbrs_data = nbrs_data[i]

        return nbrs_data

    def check_mindex(self,mindex):
        m = self.meds

        #check that current gal has OK stamp, or return bad crap
        if (m['orig_start_row'][mindex,0] == -9999
                or m['orig_start_col'][mindex,0] == -9999
                or self.is_masked[mindex]):

            nbr_numbers = np.array([-1],dtype=int)
            return nbr_numbers

        q, = np.where(
            (self.l[mindex] < self.r)
            &
            (self.r[mindex] > self.l)
        )
        if q.size > 0:
            qt, = np.where(
                (self.t[mindex] > self.b[q])
                &
                (self.b[mindex] < self.t[q])
            )
            q = q[qt]
            if q.size > 0:
               # remove dups and crap
               qt, = np.where(
                   (m['number'][mindex] != m['number'][q])
                   &
                   (m['orig_start_row'][q,0] != -9999)
                   & (m['orig_start_col'][q,0] != -9999)
               )
               q = q[qt]

        nbr_numbers = m['number'][q]
        if nbr_numbers.size > 0:
            nbr_numbers = np.unique(nbr_numbers)
            inds = nbr_numbers-1
            q, = np.where(
                  (m['orig_start_row'][inds,0] != -9999)
                & (m['orig_start_col'][inds,0] != -9999)
                & (self.is_unmasked[inds])
            )
            nbr_numbers = nbr_numbers[q]


        #if have stuff return unique else return -1
        if nbr_numbers.size == 0:
            nbr_numbers = np.array([-1],dtype=int)

        return nbr_numbers

class NbrsFoF(object):
    def __init__(self,nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(np.unique(nbrs_data['number']))

        #records fofid of entry
        self.linked = np.zeros(self.Nobj,dtype='i8')
        self.fofs = {}

        self._fof_data = None

    def get_fofs(self,verbose=True):
        self._make_fofs(verbose=verbose)
        return self._fof_data

    def _make_fofs(self,verbose=True):
        #init
        self._init_fofs()


        for i in prange(self.Nobj):
            self._link_fof(i)

        for fofid,k in enumerate(self.fofs):
            inds = np.array(list(self.fofs[k]),dtype=int)
            self.linked[inds[:]] = fofid
        self.fofs = {}

        self._make_fof_data()

    def _link_fof(self,mind):
        #get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))

        #always make a base fof
        if self.linked[mind] == -1:
            fofid = copy.copy(mind)
            self.fofs[fofid] = set([mind])
            self.linked[mind] = fofid
        else:
            fofid = copy.copy(self.linked[mind])

        #loop through nbrs
        for nbr in nbrs:
            if self.linked[nbr] == -1 or self.linked[nbr] == fofid:
                #not linked so add to current
                self.fofs[fofid].add(nbr)
                self.linked[nbr] = fofid
            else:
                #join!
                self.fofs[self.linked[nbr]] |= self.fofs[fofid]
                del self.fofs[fofid]
                fofid = copy.copy(self.linked[nbr])
                inds = np.array(list(self.fofs[fofid]),dtype=int)
                self.linked[inds[:]] = fofid

    def _make_fof_data(self):
        self._fof_data = []
        for i in range(self.Nobj):
            self._fof_data.append((self.linked[i],i+1))
        self._fof_data = np.array(self._fof_data,dtype=[('fofid','i8'),('number','i8')])
        i = np.argsort(self._fof_data['number'])
        self._fof_data = self._fof_data[i]
        assert np.all(self._fof_data['fofid'] >= 0)

    def _init_fofs(self):
        self.linked[:] = -1
        self.fofs = {}

    def _get_nbrs_index(self,mind):
        q, = np.where((self.nbrs_data['number'] == mind+1) & (self.nbrs_data['nbr_number'] > 0))
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []

def plot_fofs(m,
              fof,
              orig_dims=None,
              type='dot',
              fof_type='dot',
              fof_size=1,
              minsize=2,
              show=False,
              width=1000,
              seed=None,
              plotfile=None):
    """
    make an ra,dec plot of the FOF groups

    Only groups with at least two members ares shown
    """
    import random
    random.seed(seed)
    try:
        import biggles
        import esutil as eu
        have_biggles=True
    except ImportError:
        have_biggles=False
        
    if not have_biggles:
        print("skipping FOF plot because biggles is not "
              "available")
        return

    x = m['ra']
    y = m['dec']

    hd=eu.stat.histogram(fof['fofid'], more=True)
    wlarge,=np.where(hd['hist'] >= minsize)
    ngroup=wlarge.size
    if ngroup > 0:
        colors=rainbow(ngroup)
        random.shuffle(colors)
    else:
        colors=None

    print("unique groups >= %d: %d" % (minsize,wlarge.size))
    print("largest fof:",hd['hist'].max())

    xmin,xmax = x.min(), x.max()
    ymin,ymax = y.min(), y.max()
    if orig_dims is not None:
        xmin,xmax=0,orig_dims[1]
        ymin,ymax=0,orig_dims[0]
        xrng=[xmin,xmax]
        yrng=[ymin,ymax]
        aratio = (ymax-ymin)/(xmax-xmin)
    else:
        xrng,yrng=None,None
        #aratio=None
        aratio = (ymax-ymin)/(xmax-xmin)

    plt=biggles.FramedPlot(
        xlabel='RA',
        ylabel='DEC',
        xrange=xrng,
        yrange=yrng,
        aspect_ratio=aratio,
    )

    allpts=biggles.Points(
        x, y,
        type=type,
    )
    plt.add(allpts)

    rev=hd['rev']
    icolor=0
    for i in range(hd['hist'].size):
        if rev[i] != rev[i+1]:
            w=rev[ rev[i]:rev[i+1] ]
            if w.size >= minsize:
                indices=fof['number'][w]-1

                color=colors[icolor]
                xx=np.array(x[indices],ndmin=1)
                yy=np.array(y[indices],ndmin=1)

                pts = biggles.Points(
                    xx, yy, 
                    type=fof_type,
                    size=fof_size,
                    color=color,
                )

                plt.add(pts)
                icolor += 1

    height=int(width*aratio)
    if plotfile is not None:
        ffront=os.path.basename(plotfile)
        name=ffront.split('-mof-')[0]
        plt.title='%s FOF groups' % name

        print("writing:",plotfile)
        plt.write_img(width,int(height),plotfile)

    if show:
        plt.show(width=width, height=height)

def rainbow(num, type='hex'):
    """
    make rainbow colors

    parameters
    ----------
    num: integer
        number of colors
    type: string, optional
        'hex' or 'rgb', default hex
    """
    import colorsys

    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    # not going to 360
    minh = 0.0
    # 270 would go to pure blue
    #maxh = 270.0
    maxh = 285.0

    if num==1:
        hstep=0
    else:
        hstep = (maxh-minh)/(num-1)

    colors=[]
    for i in range(num):
        h = minh + i*hstep

        # just change the hue
        r,g,b = colorsys.hsv_to_rgb(h/360.0, 1.0, 1.0)
        r *= 255
        g *= 255
        b *= 255
        if type == 'rgb':
            colors.append( (r,g,b) )
        elif type == 'hex':

            rgb = (int(r), int(g), int(b))
            colors.append( rgb_to_hex(rgb) )
        else:
            raise ValueError("color type should be 'rgb' or 'hex'")

    return colors


