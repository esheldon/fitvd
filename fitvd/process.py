"""
TODO:

    - tweak group finding
    - deal with fact that des is coadd only and vista has no coadd
    - deal with masking (some vista are all zeros)

"""
import numpy as np
import logging
import time
import ngmix
import ngmix.medsreaders
import fitsio
import yaml
import esutil as eu
import meds

from . import fitting
from . import files
from . import vis
from . import util
from . import desbits
from . import procflags
from .procflags import get_flagname
from . import fofs

logger = logging.getLogger(__name__)

class Processor(object):
    """
    class to process a set of observations.
    """
    def __init__(self, args):
        self.args=args

        self._set_rng()
        self._set_blacklist()
        self._load_conf()
        self._load_meds_files()
        self._load_fofs()
        self._set_fof_range()
        self._set_fitter()

    def go(self):
        """
        process the requested FoF groups
        """
        olist=[]
        elist=[]
        flist=[]

        tm0 = time.time()
        nfofs = self.end-self.start+1

        for fofid in range(self.start,self.end+1):
            logger.info('processing: %d:%d' % (fofid,self.end))

            output, epochs_data, fof_data = self._process_fof(fofid)
            flist.append(fof_data)

            if output is not None:
                olist.append(output)
                if epochs_data is not None:
                    elist.append(epochs_data)


        fof_data = eu.numpy_util.combine_arrlist(flist)

        output = eu.numpy_util.combine_arrlist(olist)
        if len(elist) > 0:
            epochs_data = eu.numpy_util.combine_arrlist(elist)
        else:
            epochs_data = None

        tm = time.time()-tm0
        logger.info('total time: %g' % tm)
        logger.info('time per: %g' % (tm/nfofs))

        self._write_output(output, epochs_data, fof_data)

    def _process_fof(self, fofid):
        """
        process single FoF group
        """

        tp = time.time()
        fof_data = self._get_fof_struct()

        w,=np.where(self.fofs['fofid'] == fofid)
        fof_size = w.size
        logger.info('FoF size: %d' % fof_size)
        assert w.size > 0,'no objects found for FoF id %d' % fofid

        maxs = self.config['max_fof_size']
        if fof_size > maxs:
            logger.info('skipping FoF group with '
                        'size %d > %d' % (fof_size, maxs))
            skip_fit = True
        else:
            skip_fit = False

        # assumes meds is number sorted
        indices = self.fofs['number'][w]-1

        logger.debug('loading data')
        mbobs_list, flags = self._get_fof_mbobs_list(indices)
        if flags != 0:
            output, epochs_data = self._get_empty_output(indices)
            output['flags'] = flags
            output['flagstr'] = get_flagname(flags)
        else:

            if self.args.save or self.args.show:
                self._doplots(fofid, mbobs_list)

            logger.debug('doing fits')
            output, epochs_data = self.fitter.go(
                mbobs_list,
                ntry=self.config['mof']['ntry'],
                skip_fit=skip_fit,
            )
            if skip_fit:
                output['flags'] = procflags.FOF_TOO_LARGE
                output['flagstr'] = procflags.get_flagname(procflags.FOF_TOO_LARGE)


        logger.info('fit result: %s' % get_flagname(output['flags'][0]) )

        self._add_extra_outputs(indices, output, fofid, self.fofs[w])

        tp = time.time()-tp
        fof_data['fof_id'] = fofid
        fof_data['fof_size'] = w.size
        fof_data['fof_time'] = tp
        logger.info('FoF time: %g' % tp)

        if mbobs_list is not None and (self.args.save or self.args.show):
            self._doplots_compare_model(fofid, output, mbobs_list)

        return output, epochs_data, fof_data

    def _get_fof_struct(self):
        dt=[
            ('fof_id','i8'),
            ('fof_size','i4'),
            ('fof_time','f4'),
        ]
        return np.zeros(1, dtype=dt)

    def _get_empty_output(self, indices):
        nobj = indices.size
        output = self.fitter._get_struct(nobj)
        epochs_data = self.fitter._get_epochs_struct()
        return output, epochs_data

    def _add_extra_outputs(self, indices, output, fofid, fofs):

        m = self.mb_meds.mlist[0]
        output['id'] = m['id'][indices]
        output['ra'] = m['ra'][indices]
        output['dec'] = m['dec'][indices]
        output['fof_id'] = fofid
        output['fof_size'] = output.size
        output['mask_flags'] = fofs['mask_flags']

    def _get_fof_mbobs_list(self, indices):
        """
        load the mbobs_list for the input FoF group list
        """
        mbobs_list=[]
        for index in indices:
            mbobs, flags = self._get_mbobs(index)
            if flags != 0:
                return None, flags

            mbobs_list.append(mbobs)

        return mbobs_list, 0

    def _cut_high_maskfrac(self, mbobs):
        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        flags = 0
        mf = self.config['max_maskfrac']
        for band,obslist in enumerate(mbobs):
            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for epoch,obs in enumerate(obslist):
                npix, nmasked = util.get_badpix_frac_sums(obs)
                maskfrac = nmasked/npix
                if maskfrac < mf:
                    new_obslist.append(obs)
                else:
                    logger.info(
                        'cutting cutout band %d '
                        'epoch %d for maskfrac: %g' % (band,epoch,maskfrac)
                    )

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                flags = procflags.HIGH_MASKFRAC

            new_mbobs.append(new_obslist)

        return new_mbobs, flags

    def _cut_masked_center(self, mbobs):
        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        flags = 0
        cconf = self.config['cut_masked_center']
        rad = cconf['radius_pixels']

        id = mbobs[0][0].meta['id']

        for band,obslist in enumerate(mbobs):
            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for epoch,obs in enumerate(obslist):

                shape = obs.image.shape
                row, col = (np.array(shape)-1.0)/2.0

                row_start = _clip_pixel(row-rad, shape[0])
                row_end   = _clip_pixel(row+rad, shape[0])
                col_start = _clip_pixel(col-rad, shape[1])
                col_end   = _clip_pixel(col+rad, shape[1])

                wt_sub = obs.weight[
                    row_start:row_end,
                    col_start:col_end,
                ]
                wcen=np.where(wt_sub <= 0.0)
                if wcen[0].size > 0:
                    logger.info(
                        '    id %d skipping band %d '
                        'cutout %d due masked center' % (id, band,epoch)
                    )
                else:
                    new_obslist.append(obs)

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                return None, procflags.ALL_CENTERS_MASKED

            new_mbobs.append(new_obslist)

        return new_mbobs, flags

    def _cut_low_npix(self, mbobs):
        """
        when using uberseg, the masking can be high in the weight
        map.  we use this function instead which just checks the
        raw number of pixels is above some threshold
        """
        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        min_npix = self.config['min_npix']

        flags = 0
        for band,obslist in enumerate(mbobs):
            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for epoch,obs in enumerate(obslist):
                npix, nmasked = util.get_badpix_frac_sums(obs)
                ngood = npix-nmasked

                if ngood >= min_npix:
                    new_obslist.append(obs)
                else:
                    logger.info(
                        'skipping band %d '
                        'epoch %d for npix %d < %d' % (band,epoch,ngood,min_npix)
                    )

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                flags = procflags.TOO_FEW_PIXELS

            new_mbobs.append(obslist)

        return new_mbobs, flags


    def _get_mbobs(self, index):
        if self.config['weight_type'] == 'circular-mask':
            weight_type='weight'
        else:
            weight_type=self.config['weight_type']

        mbobs=self.mb_meds.get_mbobs(
            index,
            weight_type=weight_type,
        )

        if len(mbobs) < self.mb_meds.nband:
            return None, procflags.NO_DATA
        for obslist in mbobs:
            if len(obslist)==0:
                return None, procflags.NO_DATA

        if self.config['skip_first_epoch']:
            mbobs, flags =self._remove_first_epoch(mbobs)
            if flags != 0:
                return None, flags

        if self.blacklist is not None:
            mbobs = util.check_blacklist(mbobs, self.blacklist)
            if mbobs is None:
                return None, procflags.NO_DATA

        # need to do this *before* trimming
        if self.config['reject_outliers']:
            mbobs, flags = self._reject_outliers(mbobs)
            if flags != 0:
                return None, flags

        if 'trim_images' in self.config:
            mbobs, flags = self._trim_images(mbobs, index)
            if flags != 0:
                return None, flags

        if self.config['image_flagnames_to_mask'] is not None:
            mbobs, flags = util.zero_bitmask_in_weight(mbobs, self.config['image_flagvals_to_mask'])
            if flags != 0:
                return None, flags

        if self.config['weight_type'] == 'uberseg':
            mbobs, flags = self._cut_low_npix(mbobs)
        else:
            mbobs, flags = self._cut_high_maskfrac(mbobs)

        # before setting weight, because circular masking can blank
        # out lots of the stamp but its ok
        mbobs.meta['badpix_frac'] = util.get_badpix_frac(mbobs)

        if flags != 0:
            return None, flags

        if 'inject' in self.config and self.config['inject']['do_inject']:
            self._inject_fake_objects(mbobs)

        mbobs, flags = self._set_weight(mbobs, index)
        if flags != 0:
            return None, flags

        if 'cut_masked_center' in self.config:
            mbobs, flags = self._cut_masked_center(mbobs)
            if flags != 0:
                return None, flags

        if 'flux' in self.config['parspace']:
            mname=self.config['mof']['model']
            name = '%s_pars' % mname
            mbobs.meta['input_model_pars'] = self.model_pars[name][index].copy()
            mbobs.meta['input_flags'] = self.model_pars['flags'][index].copy()
            #logger.debug('added input pars: %s' % str(mbobs.meta['input_model_pars']))

        for band,obslist in enumerate(mbobs):
            m=self.mb_meds.mlist[band]

            if hasattr(self,'offsets'):
                if len(mbobs)==1:
                    voffset = self.offsets['voffset'][index]
                    uoffset = self.offsets['uoffset'][index]
                else:
                    voffset = self.offsets['voffset'][index, band]
                    uoffset = self.offsets['uoffset'][index, band]

                for obs in obslist:
                    jac = obs.jacobian
                    row,col = jac.get_rowcol(voffset, uoffset)
                    jac.set_cen(row=row, col=col)
                    obs.set_jacobian(jac)

            scale = obslist[0].jacobian.scale

            meta={'magzp_ref': self.magzp_refs[band]}

            radcol = self.config['radius_column']
            cat = m.get_cat()
            if radcol in cat.dtype.names:
                rad = m[radcol][index]

                if 'arcsec' not in radcol:
                    scale = mbobs[0][0].jacobian.get_scale()
                    rad = rad*scale
                    meta['Tsky'] = 2* (rad*0.5)**2

            obslist.meta.update(meta)

            # fudge for ngmix working in surface brightness
            if self.config['parspace']=='ngmix':
                for obs in obslist:
                    pixel_scale2 = obs.jacobian.get_scale()**2
                    pixel_scale4 = pixel_scale2*pixel_scale2
                    obs.image *= 1/pixel_scale2
                    obs.weight *= pixel_scale4

        return mbobs, 0

    def _remove_first_epoch(self, mbobs):
        """
        remove first "epoch" from all Obslist.  This is usually
        to skip the coadd
        """
        logging.debug('skipping first "epoch"')
        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update(mbobs.meta)

        for obslist in mbobs:

            logging.debug('starting nepoch: %d' % len(obslist))
            if len(obslist)==1:
                return None, procflags.NO_DATA

            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for obs in obslist[1:]:
                new_obslist.append(obs)

            logging.debug('ending nepoch: %d' % len(new_obslist))
            new_mbobs.append(new_obslist)

        return new_mbobs, 0

    def _reject_outliers(self, mbobs):
        """
        remove first "epoch" from all Obslist.  This is usually
        to skip the coadd
        """
        logging.debug('rejecting outliers')

        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )


        for band,obslist in enumerate(mbobs):

            imlist=[]
            wtlist = []
            for obs in obslist:
                imlist.append(obs.image)
                wtlist.append(obs.weight)

            nreject=meds.reject_outliers(imlist,wtlist)
            if nreject > 0:
                id=obslist[0].meta['id']
                logger.info(
                    '    id %d band %d rejected %d outlier pixels' % (id,band,nreject)
                )

                new_obslist = ngmix.ObsList()
                new_obslist.meta.update(obslist.meta)
                for obs in obslist:
                    # this will force an update of the pixels list
                    try:
                        obs.update_pixels()
                        new_obslist.append( obs )
                    except ngmix.GMixFatalError as err:
                        # all pixels are masked
                        pass
            else:
                new_obslist = obslist

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                flags = procflags.HIGH_MASKFRAC
                return None, flags

            new_mbobs.append(new_obslist)

        return new_mbobs, 0

    def _sample_fake(self, conf):
        if isinstance(conf,dict):
            if conf['type']=='uniform':
                rng=conf['range']
                val = self.rng.uniform(
                    low=rng[0],
                    high=rng[1],
                )
            elif conf['type']=='log-uniform':
                rng=conf['range']
                logrng = [np.log10(rng[0]), np.log10(rng[1])]
                logvals = self.rng.uniform(
                    low=logrng[0],
                    high=logrng[1],
                )
                val = 10.0**logvals

        else:
            val=conf

        return val



    def _inject_fake_objects(self, mbobs):
        """
        inject a simple model for quick tests
        """
        import galsim

        iconf=self.config['inject']

        model_name=iconf['model']

        flux = self._sample_fake(iconf['flux'])
        hlr = self._sample_fake(iconf['hlr'])

        if model_name=='exp':
            model0 = galsim.Exponential(
                half_light_radius=hlr,
                flux=flux,
            )

        elif model_name=='bdf':

            fracdev = self._sample_fake(iconf['fracdev'])

            eobj = galsim.Exponential(
                half_light_radius=hlr,
                flux=flux*(1-fracdev),
            )
            dobj = galsim.DeVaucouleurs(
                half_light_radius=hlr*iconf['dev_hlr_ratio'],
                flux=flux*fracdev,
            )

            model0 = galsim.Add(eobj, dobj)
        else:
            raise ValueError('bad model: "%s"' % model_name)

        if 'psf' in iconf:
            psf_model = galsim.Gaussian(
                fwhm=iconf['psf']['fwhm'],
            )
            method='fft'
        else:
            psf_model=None
            method='no_pixel'

        Tfake = ngmix.moments.fwhm_to_T(hlr/0.5)

        for obslist in mbobs:
            obslist.meta['Tsky'] = Tfake
            for obs in obslist:

                jac = obs.jacobian
                ccen = (np.array(obs.image.shape)-1.0)/2.0
                voffset,uoffset = jac.get_vu(ccen[0], ccen[1])
                model = model0.shift(-uoffset, -voffset)

                gsimage = galsim.Image(
                    obs.image.copy(),
                    wcs=jac.get_galsim_wcs(),
                )

                if psf_model is None:
                    #import images
                    #images.multiview(obs.psf.image/obs.psf.image.max(),nonlinear=0.4)
                    normpsf = obs.psf.image/obs.psf.image.sum()
                    psf_gsimage = galsim.Image(
                        normpsf,
                        wcs=obs.psf.jacobian.get_galsim_wcs(),
                    )
                    psf_to_conv = galsim.InterpolatedImage(
                        psf_gsimage,
                        x_interpolant='lanczos15',
                    )
                    obs.psf.image = normpsf

                else:

                    pshape=obs.psf.image.shape
                    psf_gsimage = psf_model.drawImage(
                        nx=pshape[1],
                        ny=pshape[0],
                        wcs=obs.psf.jacobian.get_galsim_wcs(),
                    )

                    #psf_to_conv = galsim.InterpolatedImage(
                    #    psf_gsimage,
                    #)
                    psf_to_conv = psf_model
                    obs.psf.image = psf_gsimage.array

                tmodel = galsim.Convolve(
                    model,
                    psf_to_conv,
                )

                tmodel.drawImage(
                    image=gsimage,
                    method=method,
                )

                image = gsimage.array

                wtmax = obs.weight.max()
                err = np.sqrt(1.0/wtmax)

                image += self.rng.normal(
                    scale=err,
                    size=image.shape,
                )

                obs.image = image


    def _get_best_epochs(self, index, mbobs):
        """
        just keep the best epoch if there are more than one

        this is good when using coadds and more than one epoch
        means overlap
        """
        new_mbobs=ngmix.MultiBandObsList()
        new_mbobs.meta.update(mbobs.meta)

        for band,obslist in enumerate(mbobs):
            nepoch=len(obslist)
            if nepoch > 1:

                mess='    obj %d band %d keeping best of %d epochs'
                logger.debug(mess % (index, band,nepoch))

                wts=np.array([ obs.weight.sum() for obs in obslist])
                logger.debug('    weights: %s' % str(wts))
                ibest=wts.argmax()
                keep_obs = obslist[ibest]

                new_obslist=ngmix.ObsList()
                new_obslist.meta.update(obslist.meta)
                new_obslist.append(keep_obs)
            else:
                new_obslist = obslist

            new_mbobs.append(new_obslist)
        return new_mbobs

    def _extract_radius(self, band, obslist, index):
        radcol = self.config['radius_column']
        m=self.mb_meds.mlist[band]
        cat=m.get_cat()

        rad=None
        if radcol in cat.dtype.names:
            rad = m[radcol][index]*3.0
            if 'arcsec' not in radcol:
                scale = obslist[0].jacobian.get_scale()
                rad = rad*scale

        return rad

    def _trim_images(self, mbobs, index):
        """
        trim the images down to a minimal size
        """
        logger.debug('trimming')

        min_size = self.config['trim_images']['min_size']
        max_size = self.config['trim_images']['max_size']

        min_rad = min_size/2.0
        max_rad = max_size/2.0

        new_mbobs=ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        # make sure at least one has it
        radone=None
        for band, obslist in  enumerate(mbobs):
            radone = self._extract_radius(band, obslist, index)
            if radone is not None:
                break

        assert radone is not None, \
            'at least one band should have radius_column "%s"' % radcol

        for band,obslist in enumerate(mbobs):

            m=self.mb_meds.mlist[band]

            rad = self._extract_radius(band, obslist, index)
            if rad is None:
                rad = radone

            new_obslist=ngmix.ObsList()
            new_obslist.meta.update( obslist.meta )
            for obs in obslist:
                imshape=obs.image.shape
                if imshape[0] > min_size:

                    meta = obs.meta
                    jac = obs.jacobian
                    cen = jac.get_cen()
                    rowpix=int(round(cen[0]))
                    colpix=int(round(cen[1]))


                    scale = jac.scale
                    radpix = rad/scale

                    if radpix < min_rad:
                        radpix = min_rad

                    if radpix > max_rad:
                        radpix = max_rad

                    radpix = int(radpix)-1

                    row_start = rowpix-radpix
                    row_end   = rowpix+radpix+1
                    col_start = colpix-radpix
                    col_end   = colpix+radpix+1

                    if row_start < 0:
                        row_start = 0
                    if row_end > imshape[0]:
                        row_end = imshape[0]
                    if col_start < 0:
                        col_start = 0
                    if col_end > imshape[1]:
                        col_end = imshape[1]


                    subim = obs.image[
                        row_start:row_end,
                        col_start:col_end,
                    ]
                    subwt = obs.weight[
                        row_start:row_end,
                        col_start:col_end,
                    ]
                    logger.debug('%s -> %s' % ( str(obs.image.shape),str(subim.shape)))

                    cen = (cen[0] - row_start, cen[1] - col_start)
                    jac.set_cen(row=cen[0], col=cen[1])

                    meta['orig_start_row'] += row_start
                    meta['orig_start_col'] += col_start

                    try:
                        new_obs = ngmix.Observation(
                            subim,
                            weight=subwt,
                            jacobian=jac,
                            meta=obs.meta,
                            psf=obs.psf,
                        )
                    except ngmix.GMixFatalError as err:
                        new_obs = None
                else:
                    new_obs = obs

                if new_obs is not None:
                    new_obslist.append(new_obs)

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                flags = procflags.HIGH_MASKFRAC
                return None, flags

            new_mbobs.append(new_obslist)

        return new_mbobs, 0



    def _set_weight(self, mbobs, index):
        """
        set the weight

        we set a circular mask based on the radius.  For non hst bands
        we add quadratically with a fake psf fwhm of 1.5 arcsec
        """


        if self.config['weight_type'] in ('weight', 'uberseg'):
            return mbobs, 0

        assert self.config['weight_type'] in ('circular-mask',)

        # extra space around the object
        fwhm=1.5
        sigma=fwhm/2.35
        exrad=3*sigma

        # not all meds files will have the radius column
        radcol = self.config['radius_column']
        radlist = []
        for band,obslist in enumerate(mbobs):
            m=self.mb_meds.mlist[band]
            cat = m.get_cat()
            if radcol in cat.dtype.names:
                rad = cat[radcol][index]
                if 'arcsec' not in radcol:
                    scale = m.get_obs(0,0).jacobian.scale
                    rad = rad*scale
                radlist.append( rad )

        assert len(radlist) > 0,'expected radius in one meds at least'
        radius_arcsec = max(radlist)
        radius_arcsec = np.sqrt(radius_arcsec**2 + exrad**2)

        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        for band,obslist in enumerate(mbobs):

            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for epoch,obs in enumerate(obslist):
                imshape=obs.image.shape
                jac = obs.jacobian
                scale = jac.scale
                rad_pix = radius_arcsec/scale
                rad_pix2 = rad_pix**2

                rows, cols = np.mgrid[
                    0:imshape[0],
                    0:imshape[1],
                ]
                cen = jac.cen
                rows = rows.astype('f4') - cen[0]
                cols = cols.astype('f4') - cen[1]
                rad2 = rows**2 + cols**2
                w=np.where(rad2 > rad_pix2)

                twt = obs.weight.copy()

                if w[0].size > 0:
                    twt[w] = 0.0

                wgood = np.where(twt > 0.0)
                if wgood[0].size >= self.config['min_npix']:
                    obs.weight = twt
                    new_obslist.append(obs)
                else:
                    mess='skipping band %d epoch %d for npix %d < %d'
                    mess = mess % (band,epoch,wgood[0].size,
                                   self.config['min_npix'])
                    logger.info(mess)

            if len(new_obslist)==0:
                return None, procflags.TOO_FEW_PIXELS

            new_mbobs.append( new_obslist )

        return new_mbobs, 0

    def _doplots(self, fofid, mbobs_list):
        plt=vis.view_mbobs_list(fofid, mbobs_list, show=self.args.show, save=self.args.save)

    def _doplots_compare_model(self, fofid, output, mbobs_list):
        args=self.args
        try:
            mof_fitter=self.fitter.get_mof_fitter()
            if mof_fitter is not None:
                res=mof_fitter.get_result()
                if res['flags']==0:
                    vis.compare_models(mbobs_list, mof_fitter, fofid, output,
                                       show=args.show, save=args.save)
        except RuntimeError:
            logger.info('could not render model')

        if self.args.show:
            if 'q'==input('hit a key (q to quit): '):
                stop

    def _write_output(self, output, epochs_data, fof_data):
        """
        write the output as well as information from the epochs
        """
        logger.info('writing output: %s' % self.args.output)
        eu.ostools.makedirs_fromfile(self.args.output)
        with fitsio.FITS(self.args.output,'rw',clobber=True) as fits:
            fits.write(output, extname='model_fits')
            if epochs_data is not None:
                fits.write(epochs_data, extname='epochs_data')
            fits.write(fof_data, extname='fof_data')

    def _set_rng(self):
        """
        set the rng given the input seed
        """
        self.rng = np.random.RandomState(self.args.seed)

    def _set_blacklist(self):
        blacklist = None
        if self.args.blacklist is not None:
            blacklist = files.read_blacklist(self.args.blacklist)

        self.blacklist = blacklist

    def _load_conf(self):
        """
        load the yaml config
        """
        logger.info('loading config: %s' % self.args.config)
        with open(self.args.config) as fobj:
            self.config = yaml.load(fobj)

        self.config['skip_first_epoch'] = \
            self.config.get('skip_first_epoch',False)
        self.config['image_flagnames_to_mask'] = \
            self.config.get('image_flagnames_to_mask',None)

        self.config['max_fof_size'] = \
            self.config.get('max_fof_size', np.inf)

        if self.config['image_flagnames_to_mask'] is not None:
            self.config['image_flagvals_to_mask'] = desbits.get_flagvals(
                self.config['image_flagnames_to_mask']
            )
            logger.info('will mask: %s' % repr(self.config['image_flagnames_to_mask']))
            logger.info('combined val: %d' % (self.config['image_flagvals_to_mask']))

    def _set_fitter(self):
        """
        currently only MOF
        """

        c = self.config
        c['mof']['use_input_guesses'] = c['mof'].get('use_input_guesses', False)
        parspace = self.config['parspace']

        kw = {}
        if 'flux' in parspace or c['mof']['use_input_guesses']:
            assert self.args.model_pars is not None, \
                'for flux fitting send model pars'

            logger.info('reading model pars: %s' % self.args.model_pars)
            model_pars = fitsio.read(self.args.model_pars)

            mm, mmeds = eu.numpy_util.match(
                model_pars['id'],
                self.mb_meds.mlist[0]['id'],
            )
            assert mm.size == model_pars.size, \
                'some input pars did not match'

            self.model_pars = model_pars[mm]

            if c['mof']['use_input_guesses']:
                kw['guesses'] = self.model_pars

        if parspace=='ngmix':
            self.fitter = fitting.MOFFitter(
                self.config,
                self.mb_meds.nband,
                self.rng,
                **kw
            )
        elif parspace=='galsim':
            self.fitter = fitting.MOFFitterGS(
                self.config,
                self.mb_meds.nband,
                self.rng,
                **kw
            )
        elif parspace == 'ngmix-flux':
            self.fitter = fitting.MOFFluxFitter(
                self.config,
                self.mb_meds.nband,
                self.rng,
                **kw
            )

        elif parspace=='galsim-flux':
            self.fitter = fitting.MOFFluxFitterGS(
                self.config,
                self.mb_meds.nband,
                self.rng,
                **kw
            )

        else:
            raise ValueError('bad parspace "%s", should be '
                             '"ngmix","galsim","ngmix-flux","galsim-flux"')

    def _load_fofs(self):
        """
        load FoF group data from the input file
        """
        if self.args.fofs is None:
            cat = self.mb_meds.mlist[0].get_cat()
            logger.info('making singleton fofs')
            fofst = fofs.make_singleton_fofs(cat)
        else:
            nbrs, fofst = files.load_fofs(self.args.fofs)

        self.fofs = fofst

    def _set_fof_range(self):
        """
        set the FoF range to be processed
        """
        nfofs = self.fofs['fofid'].max()+1
        assert nfofs == np.unique(self.fofs['fofid']).size

        self.start=self.args.start
        self.end=self.args.end

        if self.start is None:
            self.start = 0

        if self.end is None:
            self.end = nfofs-1

        logger.info('processing fof range: %d:%d' % (self.start,self.end))
        if self.start < 0 or self.end >= nfofs:
            mess='FoF range: [%d,%d] out of bounds [%d,%d]'
            mess = mess % (self.start,self.end,0,nfofs-1)
            raise ValueError(mess)

    def _load_meds_files(self):
        """
        load all MEDS files
        """
        mlist=[]
        for f in self.args.meds:
            logger.info('loading meds: %s' % f)
            m = ngmix.medsreaders.NGMixMEDS(f)

            if 'psf' not in m._fits:
                m = files.MEDSPSFEx(f)

            mlist.append(m)

        self.mb_meds = ngmix.medsreaders.MultiBandNGMixMEDS(mlist)

        self.magzp_refs = []
        for m in self.mb_meds.mlist:
            meta=m.get_meta()
            #if 'magzp_ref' not in meta.dtype.names:
            #    logger.info('no magzp ref set, assuming 30.0')
            #    magzp_ref=30.0
            #else:
            #    magzp_ref = meta['magzp_ref'][0]
            magzp_ref = meta['magzp_ref'][0]
            self.magzp_refs.append(magzp_ref)

        if self.args.offsets is not None:
            logger.info('reading offsets: %s' % self.args.offsets)
            self.offsets=fitsio.read(self.args.offsets)

            s=self.offsets['voffset'].shape
            if len(s)==1:
                nband=1
            else:
                nband = s[1]

            assert nband==self.mb_meds.nband, \
                'offset nbands does not match: %d vs %d' % (nband,self.mb_meds.nband)

            mo, mmeds = eu.numpy_util.match(
                self.offsets['id'],
                mlist[0]['id'],
            )
            assert mo.size == mo.size, \
                'some offsets ids did not match'

            self.offsets = self.offsets[mo]

def _clip_pixel(pixel, npix):
    pixel=int(pixel)
    if pixel < 0:
        pixel=0
    if pixel > (npix-1):
        pixel = (npix-1)
    return pixel


