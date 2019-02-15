# fitvd
Fit DES and video data to models using the MOF deblender

Examples
=========

```bash

config=fitvd-01.yaml

# generate the friends-of-friends groups
fitvd-make-fofs \
    --conf=$config \
    --plot=fofs.png \
    --output=fofs.fits \
    SN-C3_C28_r3499p02_r_meds-Y3A2_DEEP.fits

# An example run over a subset of the FoF groups
fitvd \
    --seed=8431 \
    --config=$config \
    --output=test.fits \
    --fofs=fofs.fits \
    --start=1000 --end=1009  \
    SN-C3_C28_r3499p02_r_meds-Y3A2_DEEP.fits \
    SN-C3_C28_JHKs_meds/SN-C3_C28_r3499p02_J_meds-VIDEO_DEEP.fits
```

An example config file
```yaml

parspace: 'ngmix'

weight_type: 'circular-mask'

fofs:
    method: 'radius'
    check_seg: false

    # name in meds file, or catalog if method is catalog_radius
    radius_column: "iso_radius"

    # factor to multiply radius this happens before clipping to [min_radius,
    # max_radius] for a low threshold, the isoarea is basically covering all
    # the observed flux, so mult of 1 makes sense
    radius_mult: 1.0

    # clip the radius in pixels=sqrt(isoarea_image/pi)
    # 5 pixels is about 3 sigma for a 1'' FWHM gaussian
    # this happens after the radius_mult is applied
    min_radius_arcsec: 1.0
    max_radius_arcsec: 2.0

    # This is added to the radius. This kind of padding makes sense for radii
    # based on the iso area.  Padding happens after mult and clipping
    padding_arcsec: 0.5

    # arcsec
    #extra_psf_fwhm_arcsec: 1.0
    extra_psf_fwhm_arcsec: 0.0

mof:

    model: 'bdf'

    # for guesses
    detband: 0
   
    priors:
        cen:
            type: 'normal2d'
            sigma: 0.263

        g:
            type: 'ba'
            sigma: 0.2

        T:
            type: 'flat'
            pars: [-1.0, 1.0e+09]

        flux:
            type: 'flat'
            pars: [-10.0, 1.0e+09]

        fracdev:
            type: 'truncated-normal'
            mean: 0.5
            sigma: 0.1
            minval: -2.0
            maxval: 3.0

    psf:
        model: 'coellip3'

        ntry: 2

        lm_pars:
            maxfev: 2000
            ftol: 1.0e-5
            xtol: 1.0e-5
```

Dependencies
============

Because of the numba requirement, it is best to work in a conda environment.

Available in anaconda (I recommend using conda forge)
-----------------------------------------------------

```
scipy
numba
fitsio
pyyaml
```

Not available in anaconda 
-----------------------------------------------------

```
repo    branch
--------------
esutil  master
ngmix   master
meds    psfex2
psfex   master
mof     master
```

