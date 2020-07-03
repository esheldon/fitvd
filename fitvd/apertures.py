import numpy as np


def get_mean_aper8_flux_ratio_obslist(obslist):
    """
    get the weighted mean aper8 flux ratio
    """
    aper8_flux_ratio_sum = 0.0
    wsum = 0.0
    for obs in obslist:
        rat = get_aper8_flux_ratio(
            gmix=obs.psf.gmix,
            jacobian=obs.psf.jacobian,
        )
        twsum = obs.weight.sum()
        aper8_flux_ratio_sum += rat*twsum
        wsum += twsum

        print("ratio:", rat)
        if True or rat < 0.7:
            from espy import images
            from matplotlib import pyplot as plt
            model_im = obs.psf.gmix.make_image(obs.psf.image.shape, jacobian=obs.psf.jacobian)
            psf_r, pim = images.get_profile(obs.psf.image)
            model_r, model_pim = images.get_profile(model_im)

            # plt.scatter(psf_r, pim)
            # plt.scatter(model_r, model_pim)
            # plt.ylim(-0.001, 0.001)
            # plt.show()

            plt.ylim(1.0e-5, 0.004)
            plt.yscale("log")
            plt.scatter(psf_r, pim)
            plt.scatter(model_r, model_pim)
            plt.show()

            # plt.imshow(obs.psf.image)
            # plt.show()

    if wsum <= 0.0:
        aper8_flux_ratio = 1.0
    else:
        aper8_flux_ratio = aper8_flux_ratio_sum / wsum

    return aper8_flux_ratio


def get_aper8_flux_ratio(*, gmix, jacobian):
    """
    get ratio of flux_aper/flux within 11.11 pixels, corresponding
    to aper8 in DES

    Parameters
    ----------
    gmix: ngmix.GMix
        The mixture for which to get the aper fac
    jacobian: ngmix.Jacobian
        For making the image
    radius: float, optional
        Radius, default 11.11 pixels, like aper8 in DES
    """
    # corresponds to aper8 in DES
    radius = 11.11

    dim = 25
    cen = (dim - 1)/2

    jac_expand = jacobian.copy()
    jac_expand.set_cen(row=cen, col=cen)

    image = gmix.make_image([dim]*2, jacobian=jac_expand)

    aper_flux = get_aper_flux_expand(
        image=image,
        radius=radius,
    )

    aper_flux *= jacobian.get_scale()**2

    return aper_flux/gmix.get_flux()


def get_aper_flux_expand(*, image, radius, expand=5):
    """
    get an aperture flux.  Expand image to follow the
    method in sextractor

    Parameters
    ----------
    image: array
        The image to expand
    expand: int, optional
        Expand factor, default 5 like in SExtractor.   This doesn't matter much
    radius: float, optional
        Radius, default 11.11 pixels, like aper8 in DES

    Returns
    -------
    aperture flux
    """
    image = boost_image(image, expand)
    cen = (np.array(image.shape) - 1)/2

    rowcen = cen[0]
    colcen = cen[1]

    rows, cols = np.mgrid[
        0:image.shape[0],
        0:image.shape[1],
    ]

    rows = rows - rowcen
    cols = cols - colcen

    r2 = rows**2 + cols**2

    maxr2 = radius**2 * expand**2
    w = np.where(r2 < maxr2)
    flux = image[w].sum()

    return flux/expand**2


def boost_image(a, factor):
    """
    Resize an array to larger shape, simply duplicating values.
    """

    factor = int(factor)
    if factor < 1:
        raise ValueError("boost factor must be >= 1")

    newshape = np.array(a.shape)*factor

    slices = [
        slice(0, old, float(old)/new) for old, new in zip(a.shape, newshape)
    ]
    coordinates = np.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]
