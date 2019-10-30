"""
todo

    - when guessing from input, and fof group size 1, and no flags set,
      just copy those pars
    - support bd

"""
from __future__ import print_function

import logging
import numpy as np
from numpy import log10

import esutil as eu
import ngmix
from ngmix import format_pars
from ngmix.gexceptions import GMixRangeError

from ngmix.gexceptions import BootPSFFailure

from .util import Namer, NoDataError
from . import procflags

import mof

logger = logging.getLogger(__name__)


class FitterBase(dict):
    """
    base class for fitting
    """
    def __init__(self, conf, nband, rng):

        self.nband = nband
        self.rng = rng
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

        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep, PriorBDSep

        if 'priors' not in conf:
            return None

        ppars = conf['priors']
        if ppars.get('prior_from_mof', False):
            return None

        # center
        cp = ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        # g
        gp = ppars['g']
        assert gp['type'] == "ba"
        g_prior = self._get_prior_generic(gp)

        if 'bd' in conf['model']:
            assert 'fracdev' in ppars, \
                'set fracdev prior for bdf and bd models'
            fp = ppars['fracdev']
            fracdev_prior = self._get_prior_generic(fp)

        if conf['model'] == 'bdf' and 'flux_hlr' in ppars:
            assert self.nband == 1, 'only 1 band for now'
            fhp_fname = ppars['flux_hlr']
            flux_hlr_prior = ngmix.gmix_ndim.GMixND(file=fhp_fname)
            prior = COSMOSBDFJointPrior(
                cen_prior,
                g_prior,
                fracdev_prior,
                flux_hlr_prior,
            )
        else:
            if 'T' in ppars:
                size_prior = self._get_prior_generic(ppars['T'])
            elif 'hlr' in ppars:
                size_prior = self._get_prior_generic(ppars['hlr'])
            else:
                raise ValueError('need T or hlr in priors')

            flux_prior = self._get_prior_generic(ppars['flux'])

            if conf['model'] == 'bd':
                assert 'logTratio' in ppars, "set logTratio prior for bd model"
                logTratiop = ppars['logTratio']
                logTratio_prior = self._get_prior_generic(logTratiop)

                prior = PriorBDSep(
                    cen_prior,
                    g_prior,
                    size_prior,
                    logTratio_prior,
                    fracdev_prior,
                    [flux_prior]*self.nband,
                )

            elif conf['model'] == 'bdf':
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
        ptype = ppars['type']
        bounds = ppars.get('bounds', None)

        if ptype == "flat":
            assert bounds is None, 'bounds not supported for flat'
            prior = ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == "bounds":
            prior = ngmix.priors.LMBounds(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            assert bounds is None, 'bounds not supported for erf'
            prior = ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype == 'sinh':
            assert bounds is None, 'bounds not supported for Sinh'
            prior = ngmix.priors.Sinh(
                ppars['mean'],
                ppars['scale'],
                rng=self.rng,
            )

        elif ptype == 'normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                bounds=bounds,
                rng=self.rng,
            )

        elif ptype == 'truncated-normal':
            assert 'do not use truncated normal'
            prior = ngmix.priors.TruncatedGaussian(
                mean=ppars['mean'],
                sigma=ppars['sigma'],
                minval=ppars['minval'],
                maxval=ppars['maxval'],
                rng=self.rng,
            )

        elif ptype == 'log-normal':
            assert bounds is None, 'bounds not yet supported for LogNormal'
            if 'shift' in ppars:
                shift = ppars['shift']
            else:
                shift = None

            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                shift=shift,
                rng=self.rng,
            )

        elif ptype == 'normal2d':
            assert bounds is None, 'bounds not yet supported for Normal2D'
            prior = ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype == 'ba':
            assert bounds is None, 'bounds not supported for BA'
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior


class MOFFitter(FitterBase):
    """
    class for multi-object fitting
    """
    def __init__(self, *args, **kw):

        self.guesses = kw.pop('guesses', None)

        super(MOFFitter, self).__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])
        self._set_mof_fitter_class()
        self._set_guess_func()

    def go(self, mbobs_list, ntry=2, get_fitter=False, skip_fit=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        skip_fit: bool
            If True, only fit the psfs, skipping the main deblending
            fit

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        mofc = self['mof']
        lm_pars = mofc.get('lm_pars', None)

        dofit = True
        try:
            fit_all_psfs(mbobs_list, self['mof']['psf'])

            self._do_measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            if self.guesses is not None and len(mbobs_list) == 1:
                tid = mbobs_list[0][0][0].meta['id']
                w, = np.where(self.guesses['id'] == tid)
                assert w.size == 1
                index = w[0]
                flags = self.guesses['flags'][index]
                if flags == 0:
                    dofit = False
                    logger.info('skipping fit, using input')
                    data = self.guesses[index:index+1]

            if dofit:
                fitter = self._mof_fitter_class(
                    mbobs_list,
                    mofc['model'],
                    prior=self.mof_prior,
                    lm_pars=lm_pars,
                    use_logpars=self['mof']['use_logpars'],
                )
                if skip_fit:
                    # we use a and expect the caller to set the flag
                    res = {
                        'ntry': 0,
                        'main_flags': -1,
                        'main_flagstr': 'none',
                    }
                else:
                    for i in range(ntry):
                        logger.debug('try: %d' % (i+1))

                        guess = self._guess_func(
                            mbobs_list,
                            mofc['detband'],
                            mofc['model'],
                            self.rng,
                            prior=self.mof_prior,
                            use_logpars=self['mof']['use_logpars'],
                        )

                        fitter.go(guess)

                        res = fitter.get_result()
                        if res['flags'] == 0:
                            break

                    res['ntry'] = i+1

                    if res['flags'] != 0:
                        res['main_flags'] = procflags.OBJ_FAILURE
                        res['main_flagstr'] = \
                            procflags.get_flagname(res['main_flags'])
                    else:
                        res['main_flags'] = 0
                        res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data = None
            logger.info(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            epochs_data = None
            logger.info(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        if dofit:
            reslist = fitter.get_result_list()

            data = self._get_output(
                fitter,
                mbobs_list,
                res,
                reslist,
            )

            self._mof_fitter = fitter

        return data, epochs_data

    def _do_measure_all_psf_fluxes(self, mbobs_list):
        _measure_all_psf_fluxes(mbobs_list)

    def get_mof_fitter(self):
        """
        get the MOF fitter
        """
        return self._mof_fitter

    def _set_mof_fitter_class(self):
        self._mof_fitter_class = mof.MOFStamps

    def _set_guess_func(self):
        if self.guesses is not None:
            logger.info('using input guesses')
            self._guess_func = ParsGuesser(self.guesses, get_stamp_guesses)
        else:
            self._guess_func = get_stamp_guesses

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
        model = self['mof']['psf']['model']
        return 6*ngmix.gmix.get_model_ngauss(model)

    @property
    def namer(self):
        return Namer(front=self['mof']['model'])

    def _get_epochs_dtype(self):
        dt = [
            ('id', 'i8'),
            ('band', 'i2'),
            ('file_id', 'i4'),
            ('psf_pars', 'f8', self.npars_psf),
        ]
        return dt

    def _get_epochs_struct(self):
        dt = self._get_epochs_dtype()
        data = np.zeros(1, dtype=dt)
        data['id'] = -9999
        data['band'] = -1
        data['file_id'] = -1
        data['psf_pars'] = -9999
        return data

    def _get_epochs_output(self, mbobs_list):
        elist = []
        for mbobs in mbobs_list:
            for band, obslist in enumerate(mbobs):
                for obs in obslist:
                    meta = obs.meta
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
        nbtup = (self.nband, )

        n = self.namer
        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
            (n('flags'), 'i4'),
            (n('deblend_flags'), 'i4'),
            (n('ntry'), 'i2'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_err'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
            (n('flux'), 'f8', nbtup),
            (n('mag'), 'f8', nbtup),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nbtup),
            ('gap_flux', 'f8', nbtup),
            ('gap_flux_err', 'f8', nbtup),
            ('gap_mag', 'f8', nbtup),
            # ('spread_model', 'f8'),
            # ('spread_model_flags', 'i4'),
        ]

        if 'bd' in self['mof']['model']:
            dt += [
                (n('fracdev'), 'f8'),
                (n('fracdev_err'), 'f8'),
            ]
        if self['mof']['model'] == 'bd':
            dt += [
                (n('logTratio'), 'f8'),
                (n('logTratio_err'), 'f8'),
            ]

        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)
        st['psf_flux_flags'] = procflags.NO_ATTEMPT

        # st['spread_model_flags'] = procflags.NO_ATTEMPT

        n = self.namer
        st[n('flags')] = procflags.NO_ATTEMPT

        noset = [
            'id',
            'ra', 'dec',
            'flags', 'flagstr',
            'spread_model_flags',
            'psf_flux_flags',
            n('flags'),
            'fof_id', 'fof_size',
            'mask_flags',
        ]

        for n in st.dtype.names:
            if n not in noset:
                if 'err' in n or 'cov' in n:
                    st[n] = 9.999e9
                else:
                    st[n] = -9.999e9

        return st

    def _get_output(self, fitter, mbobs_list, main_res, reslist):

        if 'ngmix' in self['parspace']:
            weight_fwhm = self['gap']['weight_fwhm']
            dogap = True
        else:
            dogap = False

        nobj = len(mbobs_list)
        output = self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        n = self.namer
        pn = Namer(front='psf')

        if 'flags' in main_res:
            output[n('flags')] = main_res['flags']

        output[n('ntry')] = main_res['ntry']
        logger.info('ntry: %s nfev: %s' %
                    (main_res['ntry'], main_res.get('nfev', None)))

        for i in range(output.size):
            t = output[i]
            mbobs = mbobs_list[i]

            for band, obslist in enumerate(mbobs):
                meta = obslist.meta
                zp = meta['magzp_ref']

                if 'psf_flux_flags' not in meta:
                    continue

                t['psf_flux_flags'][band] = meta['psf_flux_flags']
                for name in ('flux', 'flux_err', 'flux_s2n'):
                    t[pn(name)][band] = meta[pn(name)]

                tflux = t[pn('flux')][band].clip(min=0.001)
                t[pn('mag')][band] = get_mag(tflux, zp)

        for i, res in enumerate(reslist):
            t = output[i]
            for name in ['psf_g', 'psf_T']:
                if name in res:
                    t[name] = res[name]
            for name in ['deblend_flags']:
                if name in res:
                    t[n(name)] = res[name]

        # model flags will remain at NO_ATTEMPT
        if main_res['main_flags'] == 0:

            for i, res in enumerate(reslist):
                t = output[i]
                mbobs = mbobs_list[i]

                t['badpix_frac'] = mbobs.meta['badpix_frac']

                for name, val in res.items():
                    if name == 'nband':
                        continue

                    if 'psf' in name:
                        t[name] = val
                    else:

                        nname = n(name)
                        if nname == n('flux'):
                            val = self._get_flux(val)

                        t[nname] = val

                        if 'pars_cov' in name:
                            ename = n('pars_err')
                            pars_err = np.sqrt(np.diag(val))
                            t[ename] = pars_err

                for band, obslist in enumerate(mbobs):
                    meta = obslist.meta
                    zp = meta['magzp_ref']

                    tflux = t[n('flux')][band]

                    # TODO fix flux error for log pars
                    tflux_err = t[n('flux_err')][band]

                    t[n('mag')][band] = get_mag(tflux, zp)

                    if 'ngmix' in self['parspace']:
                        gm = fitter.get_gmix(i, band=band)
                        for obs in obslist:
                            obs.set_gmix(gm)

                    if dogap:
                        gap_flux = gm.get_gaussap_flux(fwhm=weight_fwhm)

                        if tflux > 0:
                            efac = gap_flux/tflux
                        else:
                            efac = 1.0

                        t['gap_flux'][band] = gap_flux
                        t['gap_flux_err'][band] = tflux_err*efac
                        t['gap_mag'][band] = get_mag(gap_flux, zp)

                # smres = calc_spread_model(mbobs)
                # t['spread_model_flags'] = smres['flags']
                # t['spread_model'] = smres['spread_model']

                if 'pars' in res:
                    pname = 'pars'
                    perr_name = 'pars_err'
                else:
                    pname = 'flux'
                    perr_name = 'flux_err'

                try:
                    pstr = ' '.join(['%8.3g' % el for el in t[n(pname)]])
                    estr = ' '.join(['%8.3g' % el for el in t[n(perr_name)]])
                except TypeError:
                    pstr = '%8.3g' % t[n(pname)]
                    estr = '%8.3g' % t[n(perr_name)]

                logger.info('%d pars: %s' % (i, pstr))
                logger.info('%d perr: %s' % (i, estr))

        return output

    def _get_flux(self, flux):
        return flux


class MOFFluxFitter(MOFFitter):
    """
    take structural parameters from input model pars, just
    fit the flux
    """
    def __init__(self, conf, nband, rng, model_data):

        super(MOFFitter, self).__init__(conf, nband, rng)

        self.model_data = model_data
        mname = conf['mof']['model']
        name = '%s_pars' % mname
        self.model_pars = model_data[name]

        self._set_mof_fitter_class()

    def go(self, mbobs_list, get_fitter=False, skip_fit=False, **kw):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        skip_fit: bool
            If True, only fit the psfs, skipping the main deblending
            fit

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        mofc = self['mof']

        try:
            fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            input_pars, input_flags = self._get_pars(mbobs_list)

            fitter = self._mof_fitter_class(
                mbobs_list,
                mofc['model'],
                input_pars,
                flags=input_flags,
            )
            if skip_fit:
                # we use a and expect the caller to set the flag
                res = {
                    'ntry': 0,
                    'main_flags': -1,
                    'main_flagstr': 'none',
                }
            else:
                fitter.go()

                res = fitter.get_result()

                res['ntry'] = 1
                res['nfev'] = -1

                if np.any(res['flags'] != 0):
                    res['main_flags'] = procflags.OBJ_FAILURE
                    res['main_flagstr'] = \
                        procflags.get_flagname(res['main_flags'])
                else:
                    res['main_flags'] = 0
                    res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data = None
            logger.info(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            epochs_data = None
            logger.info(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        reslist = fitter.get_result_list()

        data = self._get_output(
            fitter,
            mbobs_list,
            res,
            reslist,
        )

        self._mof_fitter = fitter

        return data, epochs_data

    def _get_pars(self, mbobs_list):
        idlist = np.array([m[0][0].meta['id'] for m in mbobs_list])
        mpars, mobs = eu.numpy_util.match(self.model_data['id'], idlist)

        assert mobs.size == len(idlist)

        flags = self.model_data['flags'][mpars]
        pars = self.model_pars[mpars, :]

        return pars, flags

    def _get_prior(self, *args):
        raise ValueError('dont need a prior for MOFFlux fitting')

    def _set_guess_func(self):
        raise ValueError('dont need a guess func for MOFFlux fitting')

    def _set_mof_fitter_class(self):
        self._mof_fitter_class = mof.MOFFlux

    def _get_dtype(self):
        # npars = self.npars
        npars = self.nband
        nband = self.nband
        nbtup = (self.nband, )

        n = self.namer
        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
            (n('flags'), 'i4', nbtup),
            (n('deblend_flags'), 'i4'),
            (n('ntry'), 'i2'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_err'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('flux'), 'f8', nbtup),
            (n('mag'), 'f8', nbtup),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nbtup),
            ('gap_flux', 'f8', nbtup),
            ('gap_flux_err', 'f8', nbtup),
            ('gap_mag', 'f8', nbtup),
            # ('spread_model', 'f8'),
            # ('spread_model_flags', 'i4'),
        ]

        return dt


class MOFFitterGS(MOFFitter):
    def make_image(self, iobj, band=0, obsnum=0):
        return self._mof_fitter.make_image(
            iobj, band=band, obsnum=obsnum,
        )

    def _get_flux(self, flux):
        if self['mof']['use_logpars']:
            return 10.0**flux
        else:
            return flux

    def _do_measure_all_psf_fluxes(self, mbobs_list):
        _measure_all_psf_fluxes_gs(mbobs_list)

    def _set_mof_fitter_class(self):
        if self.get('use_kspace', False):
            self._mof_fitter_class = mof.KGSMOF
        else:
            self._mof_fitter_class = mof.GSMOF

    def _set_guess_func(self):
        self._guess_func = get_stamp_guesses_gs

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband
        nbtup = (self.nband, )

        # TODO: get psf hlr too and do ratio?
        n = self.namer
        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
            (n('flags'), 'i4'),
            (n('deblend_flags'), 'i4'),
            (n('ntry'), 'i2'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_err'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('hlr'), 'f8'),
            (n('hlr_err'), 'f8'),
            (n('flux'), 'f8', nbtup),
            (n('mag'), 'f8', nbtup),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nbtup),
        ]

        if 'bd' in self['mof']['model']:
            dt += [
                (n('fracdev'), 'f8'),
                (n('fracdev_err'), 'f8'),
            ]

        if self['mof']['model'] == 'bd':
            dt += [
                (n('logTratio'), 'f8'),
                (n('logTratio_err'), 'f8'),
            ]

        return dt


class MOFFluxFitterGS(MOFFitterGS):
    """
    deblend with morphology fixed
    """
    def __init__(self, *args, **kw):

        super(MOFFitter, self).__init__(*args, **kw)

        self._set_mof_fitter_class()
        self._set_guess_func()

    def go(self, mbobs_list, ntry=2, get_fitter=False, skip_fit=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list
        skip_fit: bool
            If True, only fit the psfs, skipping the main deblending
            fit

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        try:
            fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            mofc = self['mof']
            fitter = self._mof_fitter_class(
                mbobs_list,
                mofc['model'],
            )
            if skip_fit:
                # we use a and expect the caller to set the flag
                res = {
                    'ntry': 0,
                    'main_flags': -1,
                    'main_flagstr': 'none',
                }
            else:

                for i in range(ntry):
                    guess = self._guess_func(
                        mbobs_list,
                        self.rng,
                    )
                    fitter.go(guess)

                    res = fitter.get_result()
                    if res['flags'] == 0:
                        break

                res['ntry'] = i+1

                if res['flags'] != 0:
                    res['main_flags'] = procflags.OBJ_FAILURE
                    res['main_flagstr'] = \
                        procflags.get_flagname(res['main_flags'])
                else:
                    res['main_flags'] = 0
                    res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data = None
            logger.info(str(err))
            res = {
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            epochs_data = None
            logger.info(str(err))

            res = {
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        reslist = fitter.get_result_list()

        data = self._get_output(
            fitter,
            mbobs_list,
            res,
            reslist,
        )

        self._mof_fitter = fitter

        return data, epochs_data

    def get_npars(self):
        """
        number of pars we expect
        """
        return self.nband

    def _set_mof_fitter_class(self):
        assert self['use_kspace'] is False
        self._mof_fitter_class = mof.galsimfit.GSMOFFlux

    def _set_guess_func(self):
        self._guess_func = get_stamp_flux_guesses_gs

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband
        nbtup = (self.nband, )

        # TODO: get psf hlr too and do ratio?
        n = self.namer
        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
            (n('flags'), 'i4'),
            (n('deblend_flags'), 'i4'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_err'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('flux'), 'f8', nbtup),
            (n('mag'), 'f8', nbtup),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nbtup),
        ]

        return dt


# class PSFOnlyFitter(FitterBase):

class PSFOnlyFitter(MOFFitter):
    """
    just do psf flux fitting
    """

    def __init__(self, conf, nband, rng):
        self.nband = nband
        self.rng = rng
        self.update(conf)
        self._setup()

    def _setup(self):
        """
        set some useful values
        """
        self.npars_psf = self.get_npars_psf()

    def go(self, mbobs_list, **kw):
        """
        fit psf fluxes

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
        epochs_data = None

        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        try:
            fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            res = {
                'main_flags': 0,
                'main_flagstr': '',
            }

        except NoDataError as err:
            logger.info(str(err))
            res = {
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            logger.info(str(err))

            res = {
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        data = self._get_output(
            mbobs_list,
            res,
        )

        return data, epochs_data

    def _get_dtype(self):
        nbtup = (self.nband, )

        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
        ]

        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)
        st['psf_flux_flags'] = procflags.NO_ATTEMPT

        noset = [
            'id',
            'ra', 'dec',
            'flags', 'flagstr',
            'psf_flux_flags',
            'fof_id', 'fof_size',
            'mask_flags',
        ]

        for n in st.dtype.names:
            if n not in noset:
                if 'err' in n or 'cov' in n:
                    st[n] = 9.999e9
                else:
                    st[n] = -9.999e9

        return st

    def _get_output(self, mbobs_list, main_res):

        nobj = len(mbobs_list)
        output = self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        pn = Namer(front='psf')

        for i in range(output.size):
            t = output[i]
            mbobs = mbobs_list[i]

            t['badpix_frac'] = mbobs.meta['badpix_frac']

            for band, obslist in enumerate(mbobs):
                meta = obslist.meta
                zp = meta['magzp_ref']

                if 'psf_flux_flags' not in meta:
                    continue

                t['psf_flux_flags'][band] = meta['psf_flux_flags']
                for name in ('flux', 'flux_err', 'flux_s2n'):
                    t[pn(name)][band] = meta[pn(name)]

                tflux = t[pn('flux')][band].clip(min=0.001)
                t[pn('mag')][band] = get_mag(tflux, zp)

        return output


class PSFOnlyFitterGS(FitterBase):
    """
    just do psf flux fitting
    """

    def _setup(self):
        """
        set some useful values
        """
        pass

    def go(self, mbobs_list, **kw):
        """
        fit psf fluxes

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
        epochs_data = None

        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        try:
            _measure_all_psf_fluxes_gs(mbobs_list)
            res = {
                'main_flags': 0,
                'main_flagstr': '',
            }

        except NoDataError as err:
            logger.info(str(err))
            res = {
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            logger.info(str(err))

            res = {
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        data = self._get_output(
            mbobs_list,
            res,
        )

        return data, epochs_data

    def _get_dtype(self):
        nbtup = (self.nband, )

        dt = [
            ('id', 'i8'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('fof_id', 'i8'),  # fof id within image
            ('fof_size', 'i4'),  # fof group size
            ('mask_flags', 'i4'),  # field masking not pixel
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('badpix_frac', 'f4'),
            ('psf_flux_flags', 'i4', nbtup),
            ('psf_flux', 'f8', nbtup),
            ('psf_mag', 'f8', nbtup),
            ('psf_flux_err', 'f8', nbtup),
            ('psf_flux_s2n', 'f8', nbtup),
        ]

        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)
        st['psf_flux_flags'] = procflags.NO_ATTEMPT

        noset = [
            'id',
            'ra', 'dec',
            'flags', 'flagstr',
            'psf_flux_flags',
            'fof_id', 'fof_size',
            'mask_flags',
        ]

        for n in st.dtype.names:
            if n not in noset:
                if 'err' in n or 'cov' in n:
                    st[n] = 9.999e9
                else:
                    st[n] = -9.999e9

        return st

    def _get_output(self, mbobs_list, main_res):

        nobj = len(mbobs_list)
        output = self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        pn = Namer(front='psf')

        for i in range(output.size):
            t = output[i]
            mbobs = mbobs_list[i]

            t['badpix_frac'] = mbobs.meta['badpix_frac']

            for band, obslist in enumerate(mbobs):
                meta = obslist.meta
                zp = meta['magzp_ref']

                if 'psf_flux_flags' not in meta:
                    continue

                t['psf_flux_flags'][band] = meta['psf_flux_flags']
                for name in ('flux', 'flux_err', 'flux_s2n'):
                    t[pn(name)][band] = meta[pn(name)]

                tflux = t[pn('flux')][band].clip(min=0.001)
                t[pn('mag')][band] = get_mag(tflux, zp)

        return output


def fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter = AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()


def _measure_all_psf_fluxes(mbobs_list):
    """
    fit all psfs in the input observations
    """
    fitter = AllPSFFluxFitter(mbobs_list)
    fitter.go()


def _measure_all_psf_fluxes_gs(mbobs_list):
    """
    fit all psfs in the input observations
    """
    fitter = AllPSFFluxFitterGS(mbobs_list)
    fitter.go()


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list = mbobs_list
        self.psf_conf = psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()

                    if hasattr(self, 'last_res'):
                        guess = self.last_res
                    else:
                        guess = None

                    fitter = _fit_one_psf(psf_obs, self.psf_conf, guess=guess)

                    if fitter is not None:
                        # fitter None means we skipped the fit because the
                        # gmix was already there
                        if 'em' in self.psf_conf['model']:
                            self.last_res = psf_obs.gmix.copy()
                        else:
                            res = fitter.get_result()
                            self.last_res = res['pars']


def _fit_one_psf(obs, pconf, guess=None):
    from ngmix.em import EM_MAXITER

    if obs.has_gmix():
        logger.debug('not fitting psf, gmix already present')
        return

    Tguess = 4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss = ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner = ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )

    elif 'em' in pconf['model']:
        ngauss = ngmix.bootstrap.get_em_ngauss(pconf['model'])
        runner = ngmix.bootstrap.EMRunner(
            obs,
            Tguess,
            ngauss,
            pconf['em_pars'],
        )

    else:
        runner = ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'], guess=guess)

    psf_fitter = runner.fitter
    res = psf_fitter.get_result()
    obs.update_meta_data({'fitter': psf_fitter})

    logger.debug('res: %s' % res)

    # generally maxiter isn't a problem
    if res['flags'] == 0 or res['flags'] == EM_MAXITER:
        gmix = psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))

    return psf_fitter


class AllPSFFluxFitter(object):
    def __init__(self, mbobs_list):
        self.mbobs_list = mbobs_list

    def go(self):
        for mbobs in self.mbobs_list:
            for band, obslist in enumerate(mbobs):

                if len(obslist) == 0:
                    raise NoDataError('no data in band %d' % band)

                meta = obslist.meta

                res = self._fit_psf_flux(band, obslist)
                meta['psf_flux_flags'] = res['flags']

                for n in ('psf_flux', 'psf_flux_err', 'psf_flux_s2n'):
                    meta[n] = res[n.replace('psf_', '')]

    def _fit_psf_flux(self, band, obslist):
        fitter = ngmix.fitting.TemplateFluxFitter(
            obslist,
            do_psf=True,
            normalize_psf=True,
            cen=(0, 0),
        )
        fitter.go()

        res = fitter.get_result()

        if res['flags'] == 0 and res['flux_err'] > 0:
            res['flux_s2n'] = res['flux']/res['flux_err']
        else:
            res['flux_s2n'] = -9999.0
            # raise BootPSFFailure('failed to fit psf fluxes for '
            #                      'band %d: %s' % (band, str(res)))

        return res


class AllPSFFluxFitterGS(object):
    def __init__(self, mbobs_list):
        self.mbobs_list = mbobs_list

    def go(self):
        for mbobs in self.mbobs_list:
            for band, obslist in enumerate(mbobs):

                if len(obslist) == 0:
                    raise NoDataError('no data in band %d' % band)

                meta = obslist.meta

                res = self._fit_psf_flux(band, obslist)
                meta['psf_flux_flags'] = res['flags']

                for n in ('psf_flux', 'psf_flux_err', 'psf_flux_s2n'):
                    meta[n] = res[n.replace('psf_', '')]

    def _fit_psf_flux(self, band, obslist):
        fitter = ngmix.galsimfit.GalsimTemplateFluxFitter(
            obslist,
            normalize_psf=True,
        )
        fitter.go()

        res = fitter.get_result()

        if res['flags'] == 0 and res['flux_err'] > 0:
            res['flux_s2n'] = res['flux']/res['flux_err']
        else:
            res['flux_s2n'] = -9999.0
            # raise BootPSFFailure('failed to fit psf fluxes for '
            #                      'band %d: %s' % (band, str(res)))

        return res


def get_stamp_guesses(list_of_obs,
                      detband,
                      model,
                      rng,
                      prior=None,
                      use_logpars=False):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    assert use_logpars is False

    nband = len(list_of_obs[0])

    if model == 'bd':
        npars_per = 7+nband
    elif model == 'bdf':
        npars_per = 6+nband
    else:
        npars_per = 5+nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    wt_fwhm = 1.2
    wt_T = ngmix.moments.fwhm_to_T(wt_fwhm)
    for i, mbo in enumerate(list_of_obs):
        # detobslist = mbo[detband]
        # detmeta = detobslist.meta
        momres = get_weighted_moments(mbo, fwhm=wt_fwhm)

        # T = detmeta.get('Tsky', None)
        # if T is None:
        #    T = mbo[0][0].psf.gmix.get_T()*1.2
        # psf_T = mbo[0][0].psf.gmix.get_T()
        # T = psf_T*0.25
        # T = 0.025
        Tmeas = momres['T']
        T = 1.0/(1/Tmeas - 1/wt_T)
        if T < 0.0:
            T = 0.025

        beg = i*npars_per

        # always close guess for center
        rowguess, colguess = prior.cen_prior.sample()
        guess[beg+0] = rowguess
        guess[beg+1] = colguess

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))
        # guess[beg+4] = 4*T*(1.0 + rng.uniform(low=-0.5, high=0.5))

        if 'bd' in model:
            if hasattr(prior.fracdev_prior, 'sigma'):
                # guessing close to mean seems to be important for the
                # pathological cases of an undetected object close to another
                low = prior.fracdev_prior.mean - 0.1*prior.fracdev_prior.sigma
                high = prior.fracdev_prior.mean + 0.1*prior.fracdev_prior.sigma
                while True:
                    fracdev_guess = rng.uniform(low=low, high=high)
                    if 0 < fracdev_guess < 1:
                        break
            else:
                fracdev_guess = prior.fracdev_prior.sample()

        if model == 'bd':

            low = prior.logTratio_prior.mean - 0.1*prior.logTratio_prior.sigma
            high = prior.logTratio_prior.mean + 0.1*prior.logTratio_prior.sigma
            logTratio_guess = rng.uniform(low=low, high=high)

            guess[beg+5] = logTratio_guess
            guess[beg+6] = fracdev_guess
            flux_start = 7

        elif model == 'bdf':
            guess[beg+5] = fracdev_guess
            flux_start = 6
        else:
            flux_start = 5

        for band, obslist in enumerate(mbo):
            obslist = mbo[band]
            band_meta = obslist.meta

            # note we take out scale**2 in DES images when
            # loading from MEDS so this isn't needed
            flux = band_meta['psf_flux']
            if not np.isfinite(flux):
                flux = 1.0
            low = flux
            high = flux*2.0
            flux_guess = rng.uniform(low=low, high=high)

            guess[beg+flux_start+band] = flux_guess

        # fix fluxes
        fluxes = guess[beg+flux_start:beg+flux_start+nband]
        logic = np.isfinite(fluxes) & (fluxes > 0.0)
        wgood, = np.where(logic)
        if wgood.size != nband:
            logging.info('fixing bad flux guesses: %s' % format_pars(fluxes))
            if wgood.size == 0:
                for iband, bobs in enumerate(mbo):
                    wt = bobs[0].weight
                    maxwt = wt.max()
                    if maxwt <= 0.0:
                        maxwt = 100.0
                    psigma = np.sqrt(1.0/maxwt)
                    fluxes[iband] = rng.uniform(low=psigma, high=5*psigma)
            else:
                wbad, = np.where(~logic)
                fac = 1.0 + rng.uniform(low=-0.1, high=0.1, size=wbad.size)
                fluxes[wbad] = fluxes[wgood].mean()*fac
            logging.info('new guesses: %s' % format_pars(fluxes))

        ptup = (i, format_pars(guess[beg:beg+flux_start+band+1]))

        logger.info('guess[%d]: %s' % ptup)
    return guess


def get_stamp_flux_guesses(list_of_obs, detband, model, rng, prior=None):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband

    these are not used
    detband, model, prior

    """

    nband = len(list_of_obs[0])
    npars_per = nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i, mbo in enumerate(list_of_obs):

        beg = i*npars_per

        for band, obslist in enumerate(mbo):
            obslist = mbo[band]
            band_meta = obslist.meta

            flux = band_meta['psf_flux']

            if flux < 0.01:
                flux = 0.01

            flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+band] = flux_guess

    return guess


def get_stamp_guesses_gs(list_of_obs,
                         detband,
                         model,
                         rng,
                         logpars=False,
                         prior=None,
                         use_logpars=False):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    assert model != 'bd', 'fix guesses for gs and bd'
    nband = len(list_of_obs[0])

    if model == 'bdf':
        npars_per = 6+nband
    else:
        npars_per = 5+nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    pos_range = 0.005
    for i, mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta = detobslist.meta

        T = detmeta.get('Tsky', None)
        if T is None:
            T = mbo[0][0].psf.gmix.get_T()*1.2

        if T < 2.0e-6:
            T = 2.0e-6

        hlr = 0.5*ngmix.moments.T_to_fwhm(T)
        if hlr > 0.11:
            hlr = hlr - 0.1

        beg = i*npars_per

        # always close guess for center
        guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        # log10 of half light radius
        if use_logpars:
            log10hlr = np.log10(hlr)
            lowval, highval = log10(0.9), log10(1.1)
            guess[beg+4] = log10hlr + rng.uniform(low=lowval, high=highval)
        else:
            guess[beg+4] = hlr*(1 + rng.uniform(low=-0.1, high=0.1))

        if model == 'bdf':
            low = prior.fracdev_prior.mean - 0.1*prior.fracdev_prior.sigma
            high = prior.fracdev_prior.mean + 0.1*prior.fracdev_prior.sigma
            guess[beg+5] = rng.uniform(low=low, high=high)

            flux_start = 6
        else:
            flux_start = 5

        for band, obslist in enumerate(mbo):
            obslist = mbo[band]
            band_meta = obslist.meta

            # TODO: if we get psf flux from galsim psf then we can
            # remove the scale squared
            # scale = obslist[0].jacobian.scale
            # flux = band_meta['psf_flux']/scale**2
            flux = band_meta['psf_flux']
            if flux < 1 or not np.isfinite(flux):
                flux = 1.0
            flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            if use_logpars:
                guess[beg+flux_start+band] = np.log10(flux_guess)
            else:
                guess[beg+flux_start+band] = flux_guess

        ptup = (i, format_pars(guess[beg:beg+flux_start+band+1]))
        logger.info('guess[%d]: %s' % ptup)
    return guess


def get_stamp_flux_guesses_gs(list_of_obs, rng):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband = len(list_of_obs[0])
    npars_per = nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i, mbo in enumerate(list_of_obs):

        beg = i*npars_per

        for band, obslist in enumerate(mbo):
            obslist = mbo[band]
            scale = obslist[0].jacobian.scale
            band_meta = obslist.meta

            # TODO: if we get psf flux from galsim psf then we can
            # remove the scale squared
            flux = band_meta['psf_flux']/scale**2

            if flux < 0.01:
                flux = 0.01

            flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+band] = flux_guess

    return guess


class ParsGuesser(object):
    def __init__(self, model_pars, fallback_func):
        self.model_pars = model_pars
        self.ids = self.model_pars['id']
        self.fallback_func = fallback_func

    def __call__(self, mbobs_list, detband, model, rng, prior=None):

        assert prior is not None, 'send a prior'

        nband = len(mbobs_list[0])
        pname = '%s_pars' % model

        pars = self.model_pars[pname]

        if model == 'bdf':
            npars_per = 6+nband
        else:
            npars_per = 5+nband

        if pars.shape[1] != npars_per:
            raise ValueError('guesses have npars %d, '
                             'but need %d' % (pars.shape[1], npars_per))
        nobj = len(mbobs_list)

        npars_tot = nobj*npars_per
        guesses = np.zeros(npars_tot)

        for i, mbo in enumerate(mbobs_list):
            tid = mbo[0][0].meta['id']

            w, = np.where(self.ids == tid)
            if w.size == 0:
                raise RuntimeError('none matched id %d' % tid)

            ind = w[0]
            flags = self.model_pars['flags'][ind]

            if flags != 0:
                logger.info('using fallback guesser')
                guess = self.fallback_func(
                    [mbo],
                    detband,
                    model,
                    rng,
                    prior=prior,
                )

            else:
                guess = pars[ind, :].copy()
                # ngmix.print_pars(guess, front='guess start: ')

                off1, off2 = prior.cen_prior.sample()
                # guess[0] += off1
                # guess[1] += off2
                guess[0] = off1
                guess[1] = off2

                guess[2] = rng.uniform(low=-0.05, high=0.05)
                guess[3] = rng.uniform(low=-0.05, high=0.05)

                guess[4] = guess[4]*(1.0 + rng.uniform(low=-0.05, high=0.05))
                if model == 'bdf':
                    flux_start = 6
                    if hasattr(prior.fracdev_prior, 'sigma'):
                        # guessing close to mean seems to be important for the
                        # pathological cases of an undetected object close to
                        # another

                        psigma = prior.fracdev_prior.sigma
                        low = prior.fracdev_prior.mean - 0.1*psigma
                        high = prior.fracdev_prior.mean + 0.1*psigma
                        while True:
                            new_fracdev = rng.uniform(low=low, high=high)
                            if 0 < new_fracdev < 1:
                                break
                    else:
                        new_fracdev = prior.fracdev_prior.sample()

                    guess[5] = new_fracdev

                else:
                    flux_start = 5

                fluxes = guess[flux_start:]
                nflux = fluxes.size
                r = rng.uniform(low=-0.05, high=0.05, size=nflux)
                guess[flux_start:] = fluxes*(1.0 + r)

                # ngmix.print_pars(guess, front='guess end: ')

            start = i*npars_per
            end = (i+1)*npars_per
            guesses[start:end] = guess

        return guesses


def get_shape_guess(g1, g2, width, rng, gmax=0.9):
    """
    Get guess, making sure in range
    """

    g = np.sqrt(g1**2 + g2**2)
    if g > gmax:
        fac = gmax/g

        g1 = g1 * fac
        g2 = g2 * fac

    shape = ngmix.Shape(g1, g2)

    while True:
        try:
            g1_offset = rng.uniform(low=width, high=width)
            g2_offset = rng.uniform(low=width, high=width)
            shape_new = shape.get_sheared(g1_offset, g2_offset)
            break
        except GMixRangeError:
            pass

    return shape_new.g1, shape_new.g2


def get_mag(flux, magzp, min_flux=0.001):
    if flux < min_flux:
        flux = min_flux

    return magzp - 2.5*np.log10(flux)


def calc_spread_model(mbobs):
    """
    calculate a spread model like quantity using input models

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        obs and psf obs must have gmix set
    """
    sm = SpreadModel(mbobs)
    sm.go()
    return sm.get_result()


class SpreadModel(object):
    """
    calculate a spread model like quantity using input models

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        obs and psf obs must have gmix set
    """
    def __init__(self, mbobs):

        if not isinstance(mbobs, ngmix.MultiBandObsList):
            raise ValueError('input must be a MultiBandObsList')

        self.mbobs = mbobs

    def get_result(self):
        """
        get the result structure
        """
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')
        return self._result

    def go(self):

        mod_data_sum = 0.0
        psf_data_sum = 0.0
        mod_psf_sum = 0.0
        psf_psf_sum = 0.0

        """
        fwhm = 2.5
        T = ngmix.moments.fwhm_to_T(fwhm)
        gmwt = ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )
        """
        # sps = []
        for obslist in self.mbobs:
            for obs in obslist:
                if not obs.has_gmix():
                    raise ValueError('all obs must have a gmix set')
                if not obs.psf.has_gmix():
                    raise ValueError('all psf obs must have a gmix set')

                # this makes a copy
                psf_gmix = obs.psf.gmix
                psf_gmix.set_flux(1.0)

                gm0 = obs.gmix
                gmix = gm0.convolve(psf_gmix)
                gmix.set_flux(1.0)

                # ensure psf is centered in the same place
                row, col = gmix.get_cen()
                psf_gmix.set_cen(row, col)
                # gmwt.set_cen(row, col)

                jac = obs.jacobian
                image = obs.image
                dims = image.shape

                model_im = gmix.make_image(dims, jacobian=jac)
                pmodel_im = psf_gmix.make_image(dims, jacobian=jac)
                # gmwt_im = gmwt.make_image(dims, jacobian=jac)

                # wtim = obs.weight * gmwt_im
                wtim = obs.weight

                mod_data_sum += (model_im*wtim*image).sum()
                psf_data_sum += (pmodel_im*wtim*image).sum()

                mod_psf_sum += (model_im*wtim*pmodel_im).sum()
                psf_psf_sum += (pmodel_im*wtim*pmodel_im).sum()

                """
                if psf_data_sum > 0.0 and psf_psf_sum > 0.0:
                    spread_model = (
                        mod_data_sum/psf_data_sum - mod_psf_sum/psf_psf_sum
                    )
                    sps.append(spread_model)
                """
        """
        if len(sps) > 0:
            flags = 0
            spread_model = sum(sps)/len(sps)
        else:
            spread_model = -9999.0
            flags = 2**0
        """

        if psf_data_sum <= 0.0 or psf_psf_sum <= 0.0:
            spread_model = -9999.0
            flags = 2**0
        else:
            flags = 0

            spread_model = (
                mod_data_sum/psf_data_sum - mod_psf_sum/psf_psf_sum
            )

        self._result = {
            'flags': flags,
            'spread_model': spread_model,
        }

    def go_sx(self):
        """
        something like the sx version
        """

        mod_data_sum = 0.0
        psf_data_sum = 0.0
        mod_psf_sum = 0.0
        psf_psf_sum = 0.0

        for obslist in self.mbobs:
            for obs in obslist:
                if not obs.has_gmix():
                    raise ValueError('all obs must have a gmix set')
                if not obs.psf.has_gmix():
                    raise ValueError('all psf obs must have a gmix set')

                # this makes a copy
                psf_gmix = obs.psf.gmix
                psf_gmix.set_flux(1.0)

                # ensure psf is centered in the same place
                gm0 = obs.gmix
                row, col = gm0.get_cen()

                psf_gmix.set_cen(row, col)

                # gmix = gm0.convolve(psf_gmix)
                # gmix.set_flux(1.0)

                psf_T = psf_gmix.get_T()
                gmconv = ngmix.GMixModel(
                    [0.0, 0.0, 0.0, 0.0, psf_T, 1.0],
                    'gauss',
                )
                gmix = psf_gmix.convolve(gmconv)
                gmix.set_cen(row, col)

                jac = obs.jacobian
                image = obs.image
                dims = image.shape

                model_im = gmix.make_image(dims, jacobian=jac)
                pmodel_im = psf_gmix.make_image(dims, jacobian=jac)

                wtim = obs.weight

                mod_data_sum += (model_im*wtim*image).sum()
                psf_data_sum += (pmodel_im*wtim*image).sum()

                mod_psf_sum += (model_im*wtim*pmodel_im).sum()
                psf_psf_sum += (pmodel_im*wtim*pmodel_im).sum()

        if psf_data_sum == 0.0 or psf_psf_sum == 0.0:
            spread_model = -9999.0
            flags = 2**0
        else:
            flags = 0

            spread_model = (
                mod_data_sum/psf_data_sum - mod_psf_sum/psf_psf_sum
            )
            logger.info('spread_model: %s' % spread_model)

        self._result = {
            'flags': flags,
            'spread_model': spread_model,
        }


def get_weighted_moments(mbobs, fwhm=1.2):
    T = ngmix.moments.fwhm_to_T(fwhm)
    wt = ngmix.GMixModel(
        [0.0, 0.0, 0.0, 0.0, T, 1.0],
        'gauss',
    )
    res = None
    for obslist in mbobs:
        for obs in obslist:
            res = wt.get_weighted_sums(obs, 1.e9, res=res)

    return ngmix.gmix.get_weighted_moments_stats(res)


class COSMOSBDFJointPrior(object):
    """
    currently only one band
    """
    def __init__(self, cen_prior, g_prior, fracdev_prior, flux_hlr_prior):
        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.fracdev_prior = fracdev_prior
        self.flux_hlr_prior = flux_hlr_prior
        self.nband = 1

        self.set_bounds()

    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """

        assert len(pars) == 7

        index = 0

        # center
        fdiff1, fdiff2 = self.cen_prior.get_fdiff(pars[0], pars[1])

        fdiff[index] = fdiff1
        index += 1

        fdiff[index] = fdiff2
        index += 1

        # g
        fdiff[index] = self.g_prior.get_fdiff(pars[2], pars[3])
        index += 1

        # fracdev
        fdiff[index] = self.fracdev_prior.get_fdiff(pars[5])
        index += 1

        # combined hlr and flux
        flux_hlr = np.zeros(2)
        flux_hlr[0] = pars[6]
        flux_hlr[1] = pars[4]

        lnp = self.flux_hlr_prior.get_lnprob_scalar(flux_hlr, **keys)
        chi2 = -2*lnp
        if chi2 < 0.0:
            chi2 = 0.0

        fdiff[index] = np.sqrt(chi2)
        index += 1

        return index

    def sample(self, n=None, **unused_keys):
        """
        Get random samples
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        samples = np.zeros((n, 6+self.nband))

        cen1, cen2 = self.cen_prior.sample(n)
        g1, g2 = self.g_prior.sample2d(n)
        fracdev = self.fracdev_prior.sample(n)

        log10flux, log10hlr = self.flux_hlr_prior.sample(n)
        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = log10hlr
        samples[:, 5] = fracdev
        samples[:, 6] = log10flux

        if is_scalar:
            samples = samples[0, :]

        return samples

    def set_bounds(self):
        """
        set possibe bounds
        """

        fracdev_bounds = self.fracdev_prior.bounds

        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
            #  (None,None),  # hlr
            (-6.0, 4.0),  # hlr
            (fracdev_bounds[0], fracdev_bounds[1]),
            (-3, 6.0),  # flux
            # (0.5,6.0),  # flux
        ]

        self.bounds = bounds
