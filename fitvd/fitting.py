import logging
import numpy as np
from numpy import array
from pprint import pprint

import esutil as eu
import ngmix
from  ngmix import format_pars
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation

from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import Namer, NoDataError
from . import procflags

import mof

logger = logging.getLogger(__name__)

class FitterBase(dict):
    """
    base class for fitting
    """
    def __init__(self, conf, nband, rng):

        self.nband=nband
        self.rng=rng
        self.update(conf)
        self._setup()

    def go(self, mbobs_list):
        """
        do measurements.  This is abstract
        """
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep

        if 'priors' not in conf:
            return None

        ppars=conf['priors']
        if ppars.get('prior_from_mof',False):
            return None

        # g
        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        if 'T' in ppars:
            size_prior = self._get_prior_generic(ppars['T'])
        elif 'hlr' in ppars:
            size_prior = self._get_prior_generic(ppars['hlr'])
        else:
            raise ValueError('need T or hlr in priors')

        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp=ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if conf['model']=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            #assert 'normal' in fp['type'],'only normal type priors supported for fracdev'

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                size_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:

            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                size_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        """
        get a prior object using the input specification
        """
        ptype=ppars['type']
        bounds = ppars.get('bounds',None)

        if ptype=="flat":
            assert bounds is None,'bounds not supported for flat'
            prior=ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype=="bounds":
            prior=ngmix.priors.LMBounds(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            assert bounds is None,'bounds not supported for erf'
            prior=ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype == 'sinh':
            assert bounds is None,'bounds not supported for Sinh'
            prior=ngmix.priors.Sinh(ppars['mean'], ppars['scale'], rng=self.rng)


        elif ptype=='normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                bounds=bounds,
                rng=self.rng,
            )

        elif ptype=='truncated-normal':
            assert 'do not use truncated normal'
            prior = ngmix.priors.TruncatedGaussian(
                mean=ppars['mean'],
                sigma=ppars['sigma'],
                minval=ppars['minval'],
                maxval=ppars['maxval'],
                rng=self.rng,
            )

        elif ptype=='log-normal':
            assert bounds is None,'bounds not yet supported for LogNormal'
            if 'shift' in ppars:
                shift=ppars['shift']
            else:
                shift=None
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                shift=shift,
                rng=self.rng,
            )


        elif ptype=='normal2d':
            assert bounds is None,'bounds not yet supported for Normal2D'
            prior=ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='ba':
            assert bounds is None,'bounds not supported for BA'
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior

class MOFFitter(FitterBase):
    """
    class for multi-object fitting
    """
    def __init__(self, *args, **kw):

        super(MOFFitter,self).__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])
        self._set_mof_fitter_class()
        self._set_guess_func()

    def go(self, mbobs_list, ntry=2, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list,list):
            mbobs_list=[mbobs_list]

        mofc = self['mof']
        lm_pars = mofc.get('lm_pars',None)

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            fitter = self._mof_fitter_class(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
                lm_pars=lm_pars,
            )
            for i in range(ntry):
                guess=self._guess_func(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                    prior=self.mof_prior,
                )
                #logger.debug('guess: %s' % ' '.join(['%g' % e for e in guess]))
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

            res['ntry'] = i+1

            if res['flags'] != 0:
                res['main_flags'] = procflags.OBJ_FAILURE
                res['main_flagstr'] = procflags.get_flagname(res['main_flags'])
            else:
                res['main_flags'] = 0
                res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.NO_DATA,
                'main_flagstr':procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter=None
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.PSF_FAILURE,
                'main_flagstr':procflags.get_flagname(procflags.PSF_FAILURE),
            }

        nobj = len(mbobs_list)

        if res['main_flags'] != 0:
            reslist=None
        else:
            reslist=fitter.get_result_list()

        data=self._get_output(
            mbobs_list,
            res,
            reslist,
        )

        self._mof_fitter=fitter

        return data, epochs_data

    def get_mof_fitter(self):
        """
        get the MOF fitter
        """
        return self._mof_fitter

    def _set_mof_fitter_class(self):
        self._mof_fitter_class=mof.MOFStamps

    def _set_guess_func(self):
        self._guess_func=get_stamp_guesses

    def _setup(self):
        """
        set some useful values
        """
        self.npars = self.get_npars()
        self.npars_psf = self.get_npars_psf()

    @property
    def model(self):
        """
        model for fitting
        """
        return self['mof']['model']

    def get_npars(self):
        """
        number of pars we expect
        """
        return ngmix.gmix.get_model_npars(self.model) + self.nband-1

    def get_npars_psf(self):
        model=self['mof']['psf']['model']
        return 6*ngmix.gmix.get_model_ngauss(model)

    @property
    def namer(self):
        return Namer(front=self['mof']['model'])

    def _get_epochs_dtype(self):
        dt = [
            ('id','i8'),
            ('band','i2'),
            ('file_id','i4'),
            ('psf_pars','f8',self.npars_psf),
        ]
        return dt

    def _get_epochs_struct(self):
        dt=self._get_epochs_dtype()
        data = np.zeros(1, dtype=dt)
        data['id'] = -9999
        data['band'] = -1
        data['file_id'] = -1
        data['psf_pars'] = -9999
        return data

    def _get_epochs_output(self, mbobs_list):
        elist=[]
        for mbobs in mbobs_list:
            for band, obslist in enumerate(mbobs):
                for obs in obslist:
                    meta=obs.meta
                    edata = self._get_epochs_struct()
                    edata['id'] = meta['id']
                    edata['band'] = band
                    edata['file_id'] = meta['file_id']
                    psf_gmix = obs.psf.gmix
                    edata['psf_pars'][0] = psf_gmix.get_full_pars()

                    elist.append(edata)

        edata = eu.numpy_util.combine_arrlist(elist)
        return edata

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband

        n=self.namer
        dt = [
            ('id','i8'),
            ('ra','f8'),
            ('dec','f8'),
            ('fof_id','i8'), # fof id within image
            ('flags','i4'),
            #('flagstr','U11'),
            ('flagstr','S18'),
            ('masked_frac','f4'),
            ('psf_g','f8',2),
            ('psf_T','f8'),
            ('psf_flux_flags','i4',nband),
            ('psf_flux','f8',nband),
            ('psf_mag','f8',nband),
            ('psf_flux_err','f8',nband),
            ('psf_flux_s2n','f8',nband),
            (n('flags'),'i4'),
            (n('ntry'),'i2'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_err'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('mag'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)

        n=self.namer
        st[n('flags')] = st['flags']

        noset=['id','ra','dec',
               'flags','flagstr',n('flags')]

        for n in st.dtype.names:
            if n not in noset:
                if 'err' in n or 'cof' in n:
                    st[n] =  9.999e9
                else:
                    st[n] = -9.999e9

        return st

    def _get_output(self, mbobs_list, main_res, reslist):

        nband=self.nband
        nobj = len(mbobs_list)
        output=self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        n=self.namer
        pn=Namer(front='psf')

        if 'flags' in main_res:
            output[n('flags')] = main_res['flags']

        output[n('ntry')] = main_res['ntry']
        logger.info('ntry: %d' % main_res['ntry'])

        # model flags will remain at NO_ATTEMPT
        if main_res['main_flags'] == 0:


            for i,res in enumerate(reslist):
                t=output[i] 
                mbobs = mbobs_list[i]

                t['masked_frac'] = mbobs.meta['masked_frac']

                for band,obslist in enumerate(mbobs):
                    meta = obslist.meta

                    if nband > 1:
                        t['psf_flux_flags'][band] = meta['psf_flux_flags']
                        for name in ('flux','flux_err','flux_s2n'):
                            t[pn(name)][band] = meta[pn(name)]

                        tflux = t[pn('flux')][band].clip(min=0.001)
                        t[pn('mag')][band] = meta['magzp_ref']-2.5*np.log10(tflux)


                    else:
                        t['psf_flux_flags'] = meta['psf_flux_flags']
                        for name in ('flux','flux_err','flux_s2n'):
                            t[pn(name)] = meta[pn(name)]

                        tflux = t[pn('flux')].clip(min=0.001)
                        t[pn('mag')] = meta['magzp_ref']-2.5*np.log10(tflux)



                for name,val in res.items():
                    if name=='nband':
                        continue

                    if 'psf' in name:
                        t[name] = val
                    else:
                        nname=n(name)
                        t[nname] = val

                        if 'pars_cov' in name:
                            ename=n('pars_err')
                            pars_err=np.sqrt(np.diag(val))
                            t[ename] = pars_err

                for band,obslist in enumerate(mbobs):
                    meta = obslist.meta
                    if nband > 1:
                        tflux = t[n('flux')][band].clip(min=0.001)
                        t[n('mag')][band] = meta['magzp_ref']-2.5*np.log10(tflux)
                    else:
                        tflux = t[n('flux')].clip(min=0.001)
                        t[n('mag')] = meta['magzp_ref']-2.5*np.log10(tflux)

                try:
                    pstr = ' '.join( [ '%8.3g' % el for el in t[n('pars')] ] )
                    estr = ' '.join( [ '%8.3g' % el for el in t[n('pars_err')] ] )
                except TypeError:
                    pstr = '%8.3g' % t[n('pars')]
                    estr = '%8.3g' % t[n('pars_err')]
                #logger.debug('%d pars: %s' % (i, str(t[n('pars')])))
                logger.debug('%d pars: %s' % (i, pstr))
                logger.debug('%d perr: %s' % (i, estr))

        return output


class MOFFitterGS(MOFFitter):
    def make_image(self, iobj, band=0, obsnum=0):
        return self._mof_fitter.make_image(
            iobj, band=band, obsnum=obsnum,
        )

    def _set_mof_fitter_class(self):
        if self.get('use_kspace',False):
            self._mof_fitter_class=mof.KGSMOF
        else:
            self._mof_fitter_class=mof.GSMOF

    def _set_guess_func(self):
        self._guess_func=get_stamp_guesses_gs

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband

        # TODO: get psf hlr too and do ratio?
        n=self.namer
        dt = [
            ('id','i8'),
            ('ra','f8'),
            ('dec','f8'),
            ('fof_id','i8'), # fof id within image
            ('flags','i4'),
            #('flagstr','U11'),
            ('flagstr','S18'),
            ('masked_frac','f4'),
            ('psf_g','f8',2),
            ('psf_T','f8'),
            ('psf_flux_flags','i4',nband),
            ('psf_flux','f8',nband),
            ('psf_mag','f8',nband),
            ('psf_flux_err','f8',nband),
            ('psf_flux_s2n','f8',nband),
            (n('flags'),'i4'),
            (n('ntry'),'i2'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_err'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('hlr'),'f8'),
            (n('hlr_err'),'f8'),
            #(n('hlr_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('mag'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

class MOFFluxFitterGS(MOFFitterGS):
    """
    deblend with morphology fixed
    """
    def __init__(self, *args, **kw):

        super(MOFFitter,self).__init__(*args, **kw)

        self._set_mof_fitter_class()
        self._set_guess_func()

    def go(self, mbobs_list, ntry=2, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list,list):
            mbobs_list=[mbobs_list]

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            mofc = self['mof']
            fitter = self._mof_fitter_class(
                mbobs_list,
                mofc['model'],
            )
            for i in range(ntry):
                guess=self._guess_func(
                    mbobs_list,
                    self.rng,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

            res['ntry'] = i+1

            if res['flags'] != 0:
                res['main_flags'] = procflags.OBJ_FAILURE
                res['main_flagstr'] = procflags.get_flagname(res['main_flags'])
            else:
                res['main_flags'] = 0
                res['main_flagstr'] = procflags.get_flagname(0)


        except NoDataError as err:
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.NO_DATA,
                'main_flagstr':procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter=None
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.PSF_FAILURE,
                'main_flagstr':procflags.get_flagname(procflags.PSF_FAILURE),
            }

        nobj = len(mbobs_list)


        if res['main_flags'] != 0:
            reslist=None
        else:
            reslist=fitter.get_result_list()

        data=self._get_output(
            mbobs_list,
            res,
            reslist,
        )

        self._mof_fitter=fitter

        return data, epochs_data

    def get_npars(self):
        """
        number of pars we expect
        """
        return self.nband

    def _set_mof_fitter_class(self):
        assert self['use_kspace']==False
        self._mof_fitter_class=mof.galsimfit.GSMOFFlux

    def _set_guess_func(self):
        self._guess_func=get_stamp_flux_guesses_gs

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband

        # TODO: get psf hlr too and do ratio?
        n=self.namer
        dt = [
            ('id','i8'),
            ('ra','f8'),
            ('dec','f8'),
            ('fof_id','i8'), # fof id within image
            ('flags','i4'),
            #('flagstr','U11'),
            ('flagstr','S18'),
            ('masked_frac','f4'),
            ('psf_g','f8',2),
            ('psf_T','f8'),
            ('psf_flux_flags','i4',nband),
            ('psf_flux','f8',nband),
            ('psf_mag','f8',nband),
            ('psf_flux_err','f8',nband),
            ('psf_flux_s2n','f8',nband),
            (n('flags'),'i4'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_err'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            #(n('hlr_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('mag'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        return dt


def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()

def _measure_all_psf_fluxes(mbobs_list):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFluxFitter(mbobs_list)
    fitter.go()


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list=mbobs_list
        self.psf_conf=psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    _fit_one_psf(psf_obs, self.psf_conf)

def _fit_one_psf(obs, pconf):
    Tguess=4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss=ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner=ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )

    elif 'em' in pconf['model']:
        ngauss=ngmix.bootstrap.get_em_ngauss(pconf['model'])
        runner=ngmix.bootstrap.EMRunner(
            obs,
            Tguess,
            ngauss,
            pconf['em_pars'],
        )


    else:
        runner=ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res=psf_fitter.get_result()
    obs.update_meta_data({'fitter':psf_fitter})

    if res['flags']==0:
        gmix=psf_fitter.get_gmix()
        #gmix.set_cen(0.0, 0.0)
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))

class AllPSFFluxFitter(object):
    def __init__(self, mbobs_list):
        self.mbobs_list=mbobs_list

    def go(self):
        for mbobs in self.mbobs_list:
            for band,obslist in enumerate(mbobs):

                if len(obslist) == 0:
                    raise NoDataError('no data in band %d' % band)

                meta=obslist.meta

                res = self._fit_psf_flux(band,obslist)
                meta['psf_flux_flags'] = res['flags']

                for n in ('psf_flux','psf_flux_err','psf_flux_s2n'):
                    meta[n] = res[n.replace('psf_','')]

    def _fit_psf_flux(self, band, obslist):
        fitter=ngmix.fitting.TemplateFluxFitter(
            obslist,
            do_psf=True,
        )
        fitter.go()

        res=fitter.get_result()

        if res['flags'] == 0 and res['flux_err'] > 0:
            #logger.debug('psf flux: %g +/- %g' % (res['flux'], res['flux_err']))
            res['flux_s2n'] = res['flux']/res['flux_err']
        else:
            res['flux_s2n'] = -9999.0
            raise BootPSFFailure("failed to fit psf fluxes for band %d: %s" % (band,str(res)))

        return res

def get_stamp_guesses(list_of_obs,
                      detband,
                      model,
                      rng,
                      prior=None):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband=len(list_of_obs[0])

    if model=='bdf':
        npars_per=6+nband
    else:
        npars_per=5+nband

    nobj=len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    pos_range = 0.005
    for i,mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta=detobslist.meta

        obs=detobslist[0]

        T=detmeta['Tsky']

        beg=i*npars_per

        # always close guess for center
        #guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        #guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)
        rowguess, colguess = prior.cen_prior.sample()
        guess[beg+0] = rowguess
        guess[beg+1] = colguess

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

        if model=='bdf':
            #fracdev_guess = prior.fracdev_prior.sample()
            if hasattr(prior.fracdev_prior,'sigma'):
                # guessing close to mean seems to be important for the pathological
                # cases of an undetected object close to another
                low  = prior.fracdev_prior.mean - 0.1*prior.fracdev_prior.sigma
                high = prior.fracdev_prior.mean + 0.1*prior.fracdev_prior.sigma
                while True:
                    fracdev_guess = rng.uniform(low=low, high=high)
                    if 0 < fracdev_guess < 1:
                        break
            else:
                fracdev_guess = prior.fracdev_prior.sample()

            guess[beg+5] = fracdev_guess
            flux_start=6
        else:
            flux_start=5

        for band, obslist in enumerate(mbo):
            obslist=mbo[band]
            scale = obslist[0].jacobian.scale
            band_meta=obslist.meta

            # note we take out scale**2 in DES images when
            # loading from MEDS so this isn't needed
            flux=band_meta['psf_flux']
            low = flux
            high = flux*2.0
            flux_guess=rng.uniform(low=low, high=high)
            #flux_guess=flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+flux_start+band] = flux_guess

        logger.debug('guess[%d]: %s' % (i,format_pars(guess[beg:beg+flux_start+band+1])))
    return guess

def get_stamp_guesses_gs(list_of_obs,
                         detband,
                         model,
                         rng,
                         prior=None):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband=len(list_of_obs[0])

    if model=='bdf':
        npars_per=6+nband
    else:
        npars_per=5+nband

    nobj=len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    pos_range = 0.005
    for i,mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta=detobslist.meta

        obs=detobslist[0]

        T=detmeta['Tsky']
        if T < 1.0e-6:
            T = 1.0e-6

        hlr = 0.5*ngmix.moments.T_to_fwhm(T)
        if hlr > 0.11:
            hlr = hlr - 0.1
        #hlr=0.05
        #print(hlr)
        #hlr = 0.05
        #hlr = 0.1*detmeta['flux_radius_arcsec']

        beg=i*npars_per

        # always close guess for center
        guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        # half light radius
        guess[beg+4] = hlr*(1.0 + rng.uniform(low=-0.05, high=0.05))

        if model=='bdf':
            #guess[beg+5] = rng.uniform(low=0.4,high=0.6)
            low  = prior.fracdev_prior.mean - 0.1*prior.fracdev_prior.sigma
            high = prior.fracdev_prior.mean + 0.1*prior.fracdev_prior.sigma
            guess[beg+5] = rng.uniform(low=low, high=high)

            flux_start=6
        else:
            flux_start=5

        for band, obslist in enumerate(mbo):
            obslist=mbo[band]
            scale = obslist[0].jacobian.scale
            band_meta=obslist.meta

            # TODO: if we get psf flux from galsim psf then we can
            # remove the scale squared
            flux=band_meta['psf_flux']/scale**2
            if flux < 0.01:
                flux = 0.01
            flux_guess=flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+flux_start+band] = flux_guess

    return guess

def get_stamp_flux_guesses_gs(list_of_obs, rng):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband=len(list_of_obs[0])
    npars_per = nband

    nobj=len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i,mbo in enumerate(list_of_obs):


        beg=i*npars_per

        for band, obslist in enumerate(mbo):
            obslist=mbo[band]
            scale = obslist[0].jacobian.scale
            band_meta=obslist.meta

            # TODO: if we get psf flux from galsim psf then we can
            # remove the scale squared
            flux=band_meta['psf_flux']/scale**2

            if flux < 0.01:
                flux = 0.01

            flux_guess=flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+band] = flux_guess

    return guess


