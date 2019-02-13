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

from . import fitting
from . import files
from . import vis
from . import util

logger = logging.getLogger(__name__)

class Processor(object):
    """
    class to process a set of observations.
    """
    def __init__(self, args):
        self.args=args

        self._set_rng()
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

        tm0 = time.time()
        nfofs = self.end-self.start+1

        for fofid in range(self.start,self.end+1):
            logger.info('processing: %d:%d' % (fofid,self.end))

            tp = time.time()
            output, epochs_data = self._process_fof(fofid)
            tp = time.time()-tp
            logger.info('FoF time: %g' % tp)

            if output is not None:
                olist.append(output)
                if epochs_data is not None:
                    elist.append(epochs_data)

        output = eu.numpy_util.combine_arrlist(olist)
        if len(elist) > 0:
            epochs_data = eu.numpy_util.combine_arrlist(elist)
        else:
            epochs_data = None

        tm = time.time()-tm0
        print('total time: %g' % tm)
        print('time per: %g' % (tm/nfofs))

        self._write_output(output, epochs_data)

    def _process_fof(self, fofid):
        """
        process single FoF group
        """
        w,=np.where(self.fofs['fofid'] == fofid)
        logger.info('FoF size: %d' % w.size)
        assert w.size > 0,'no objects found for FoF id %d' % fofid

        indices=self.fofs['number'][w]-1

        logger.debug('loading data')
        mbobs_list = self._get_fof_mbobs_list(indices)
        if mbobs_list is None:
            return None, None

        if self.args.save or self.args.show:
            self._doplots(fofid, mbobs_list)

        logger.debug('doing fits')
        output, epochs_data = self.fitter.go(mbobs_list)

        if self.args.save or self.args.show:
            self._doplots_compare_model(fofid, mbobs_list)

        self._add_extra_outputs(indices, output, fofid)
        return output, epochs_data

    def _add_extra_outputs(self, indices, output, fofid):

        m = self.mb_meds.mlist[0]
        output['id'] = m['id'][indices]
        output['ra'] = m['ra'][indices]
        output['dec'] = m['dec'][indices]
        output['fof_id'] = fofid

    def _get_fof_mbobs_list(self, indices):
        """
        load the mbobs_list for the input FoF group list
        """
        mbobs_list=[]
        for index in indices:
            mbobs = self._get_mbobs(index)
            if mbobs is None:
                return None
            mbobs_list.append(mbobs)

        return mbobs_list

    def _cut_high_maskfrac(self, mbobs):
        new_mbobs = ngmix.MultiBandObsList()
        new_mbobs.meta.update( mbobs.meta )

        mf = self.config['max_maskfrac']
        ok=True
        for obslist in mbobs:
            new_obslist = ngmix.ObsList()
            new_obslist.meta.update(obslist.meta)

            for obs in obslist:
                npix, nmasked = util.get_masked_frac_sums(obs)
                maskfrac = nmasked/npix
                if maskfrac > mf:
                    logger.info(
                        'cutting cutout band %d '
                        'obs %d for maskfrac: %g' % (band,icut,maskfrac)
                    )
                    new_obslist.append(obs)

            if len(new_obslist) == 0:
                logger.info('no cutouts left for band %d' % band)
                ok=False

        return new_mbobs, ok

    def _get_mbobs(self, index):
        mbobs=self.mb_meds.get_mbobs(
            index,
            weight_type='weight',
        )

        mbobs, ok = self._cut_high_maskfrac(mbobs)
        if not ok:
            return None

        if 'inject' in self.config and self.config['inject']['do_inject']:
            self._inject_fake_objects(mbobs)

        if 'trim_images' in self.config and self.config['trim_images']['trim']:
            mbobs = self._trim_images(mbobs, index)

        self._set_weight(mbobs, index)

        mbobs.meta['masked_frac'] = util.get_masked_frac(mbobs)

        if 'flux' in self.config['parspace']:
            mname=self.config['mof']['model']
            name = '%s_pars' % mname
            mbobs.meta['input_model_pars'] = self.model_pars[name][index].copy()
            mbobs.meta['input_flags'] = self.model_pars['flags'][index].copy()
            #logger.debug('added input pars: %s' % str(mbobs.meta['input_model_pars']))

        for band,obslist in enumerate(mbobs):
            m=self.mb_meds.mlist[band]

            if hasattr(self,'offsets'):
                #print('doing offsets')
                if len(mbobs)==1:
                    voffset = self.offsets['voffset'][index]
                    uoffset = self.offsets['uoffset'][index]
                else:
                    voffset = self.offsets['voffset'][index, band]
                    uoffset = self.offsets['uoffset'][index, band]

                #print('offsets:',voffset,uoffset)
                for obs in obslist:
                    jac = obs.jacobian
                    row,col = jac.get_rowcol(voffset, uoffset)
                    jac.set_cen(row=row, col=col)
                    obs.set_jacobian(jac)

            scale = obslist[0].jacobian.scale

            meta={'magzp_ref': self.magzp_refs[band]}

            radcol = self.config['fofs']['radius_column']
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
                    pixel_scale2 = obs.jacobian.get_det()
                    pixel_scale4 = pixel_scale2*pixel_scale2
                    obs.image *= 1/pixel_scale2
                    obs.weight *= pixel_scale4

        return mbobs

    def _inject_fake_objects(self, mbobs):
        """
        inject a simple model for quick tests
        """
        import galsim

        iconf=self.config['inject']

        model_name=iconf['model']
        hlr=iconf['hlr']
        flux=iconf['flux']

        if model_name=='exp':
            model = galsim.Exponential(
                half_light_radius=hlr,
                flux=flux,
            )

        elif model_name=='bdf':

            fracdev=iconf['fracdev'] 
            model = galsim.Add(
                galsim.Exponential(
                    half_light_radius=hlr,
                    flux=(1-fracdev),
                ),
                galsim.DeVaucouleurs(
                    half_light_radius=hlr,
                    flux=fracdev,
                )
            ).withFlux(flux)
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

                gsimage = galsim.Image(
                    obs.image.copy(),
                    wcs=obs.jacobian.get_galsim_wcs(),
                )

                if psf_model is None:
                    psf_gsimage = galsim.Image(
                        obs.psf.image/obs.psf.image.sum(),
                        wcs=obs.psf.jacobian.get_galsim_wcs(),
                    )
                    psf_to_conv = galsim.InterpolatedImage(
                        psf_gsimage,
                        #x_interpolant='lanczos15',
                    )
                    obs.psf.image = psf_gsimage.array

                else:

                    pshape=obs.psf.image.shape
                    psf_gsimage = psf_model.drawImage(
                        nx=pshape[1],
                        ny=pshape[0],
                        wcs=obs.psf.jacobian.get_galsim_wcs(),
                    )

                    psf_to_conv = galsim.InterpolatedImage(
                        psf_gsimage,
                    )
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
        for band,obslist in enumerate(mbobs):

            m=self.mb_meds.mlist[band]
            rad = m['iso_radius_arcsec'][index]*3.0

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

                    new_obs = ngmix.Observation(
                        subim,
                        weight=subwt,
                        jacobian=jac,
                        meta=obs.meta,
                        psf=obs.psf,
                    )
                else:
                    new_obs = obs

                new_obslist.append(new_obs)

            new_mbobs.append(new_obslist)

        return new_mbobs



    def _set_weight(self, mbobs, index):
        """
        set the weight

        we set a circular mask based on the radius.  For non hst bands
        we add quadratically with a fake psf fwhm of 1.5 arcsec
        """

        assert self.config['weight_type'] in ('weight','circular-mask')

        if self.config['weight_type'] == 'weight':
            return

        # extra space around the object
        fwhm=1.5
        sigma=fwhm/2.35
        exrad=3*sigma

        # not all meds files will have the radius column
        radcol = self.config['fofs']['radius_column']
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
        
        assert len(radlist) > 0,'expect radius in one meds at least'
        radius_arcsec = max(radlist)
        radius_arcsec = np.sqrt(radius_arcsec**2 + exrad**2)

        for obslist in mbobs:
            for obs in obslist:
                imshape=obs.image.shape
                jac = obs.jacobian
                scale = jac.scale
                rad_pix = radius_arcsec/scale
                rad_pix2 = rad_pix**2

                rows, cols = np.mgrid[
                    0:imshape[0],
                    0:imshape[1],
                ]
                #cen = (np.array(imshape)-1.0)/2.0
                cen = jac.cen
                rows = rows.astype('f4') - cen[0]
                cols = cols.astype('f4') - cen[1]
                rad2 = rows**2 + cols**2
                w=np.where(rad2 > rad_pix2)
                if w[0].size > 0:
                    twt = obs.weight.copy()
                    twt[w] = 0.0
                    obs.weight = twt

    def _doplots(self, fofid, mbobs_list):
        plt=vis.view_mbobs_list(mbobs_list, show=self.args.show, weight=True)
        if self.args.save:
            pltname='images-%06d.png' % fofid
            plt.title='FoF id: %d' % fofid
            logger.info('writing: %s' % pltname)
            plt.write(pltname,dpi=300)

    def _doplots_compare_model(self, fofid, mbobs_list):
        try:
            mof_fitter=self.fitter.get_mof_fitter()
            if mof_fitter is not None:
                res=mof_fitter.get_result()
                if res['flags']==0:
                    vis.compare_models(mbobs_list, mof_fitter)
        except RuntimeError:
            logger.info('could not render model')

        if self.args.show:
            if 'q'==input('hit a key (q to quit): '):
                stop

    def _write_output(self, output, epochs_data):
        """
        write the output as well as information from the epochs
        """
        logger.info('writing output: %s' % self.args.output)
        with fitsio.FITS(self.args.output,'rw',clobber=True) as fits:
            fits.write(output, extname='model_fits')
            if epochs_data is not None:
                fits.write(epochs_data, extname='epochs_data')

    def _set_rng(self):
        """
        set the rng given the input seed
        """
        self.rng = np.random.RandomState(self.args.seed)

    def _load_conf(self):
        """
        load the yaml config
        """
        logger.info('loading config: %s' % self.args.config)
        with open(self.args.config) as fobj:
            self.config = yaml.load(fobj)

    def _set_fitter(self):
        """
        currently only MOF
        """
        parspace = self.config['parspace']
        if parspace=='ngmix':
            self.fitter = fitting.MOFFitter(
                self.config,
                self.mb_meds.nband,
                self.rng,
            ) 
        elif parspace=='galsim':
            self.fitter = fitting.MOFFitterGS(
                self.config,
                self.mb_meds.nband,
                self.rng,
            ) 
        elif parspace=='galsim-flux':
            assert self.args.model_pars is not None, \
                'for flux fitting send model pars'

            model_pars = fitsio.read(self.args.model_pars)
            
            mm, mmeds = eu.numpy_util.match(
                model_pars['id'],
                self.mb_meds.mlist[0]['id'],
            )
            assert mm.size == model_pars.size, \
                'some input pars did not match'

            self.model_pars = model_pars[mm]

            self.fitter = fitting.MOFFluxFitterGS(
                self.config,
                self.mb_meds.nband,
                self.rng,
            ) 

        else:
            raise ValueError('bad parspace "%s", should be '
                             '"ngmix" or "galsim" or "galsim-flux"')

    def _load_fofs(self):
        """
        load FoF group data from the input file
        """
        nbrs, fofs = files.load_fofs(self.args.fofs)
        self.fofs = fofs

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
            if 'magzp_ref' not in meta.dtype.names:
                logger.info('no magzp ref set, assuming 30.0')
                magzp_ref=30.0
            else:
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
