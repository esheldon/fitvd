from __future__ import print_function

import numpy as np
import logging

logger = logging.getLogger(__name__)


def view_mbobs_list(fofid, mbobs_list, **kw):
    import biggles
    import images
    import plotting

    show = kw.get('show', False)
    save = kw.get('save', False)

    for i, mbobs in enumerate(mbobs_list):
        id = mbobs[0][0].meta['id']

        for band, obslist in enumerate(mbobs):

            grid = plotting.Grid(len(obslist))
            plt = biggles.Table(
                grid.nrow,
                grid.ncol,
            )
            aratio = grid.nrow/(grid.ncol*2)
            plt.aspect_ratio = aratio
            plt.title = 'FoF: %d id: %d band: %d' % (fofid, id, band)

            for iobs, obs in enumerate(obslist):

                im = obs.image
                wt = obs.weight

                im = im/im.max()

                row, col = grid(iobs)

                implt = images.view(im, nonlinear=0.4, show=False)
                wtplt = images.view(wt, show=False)

                tab = biggles.Table(1, 2)
                tab[0, 0] = implt
                tab[0, 1] = wtplt

                tab.title = 'id: %d band: %d obs: %d' % (id, band, iobs)
                plt[row, col] = tab

            if save:
                pltname = 'images-fof%06d-id%06d-band%d.png' % \
                    (fofid, id, band)
                logger.info('writing: %s' % pltname)
                plt.write_img(3000, 3000*aratio, pltname)

            if show:
                plt.show(width=2000, height=2000*aratio)

    return plt


def compare_models(mbobs_list, fitter, fofid, output, show=False, save=False):

    for iobj, mbobs in enumerate(mbobs_list):
        id = output['id'][iobj]
        for band, obslist in enumerate(mbobs):
            for obsnum, obs in enumerate(obslist):
                model_image = fitter.make_image(
                    iobj,
                    band=band,
                    obsnum=obsnum,
                    include_nbrs=True,
                )

                title = 'fof: %d id: %d band: %d obs: %d' % \
                    (fofid, id, band, obsnum)

                image = obs.image

                """
                plt = compare_images_mosaic(
                    image,
                    model_image,
                    labels=['image', 'model'],
                    title=title,
                    show=show,
                )
                """
                wt = obs.weight.copy()
                plt = compare_images(
                    image,
                    model_image,
                    wt,
                    labels=['image', 'model'],
                    title=title,
                    show=show,
                )

                if save:
                    fname = 'compare-fof%06d-%d-band%d-%d.png' % \
                        (fofid, id, band, obsnum)
                    print(fname)
                    # plt.write_img(1500, 1500*2.0/3.0, fname)
                    plt.write_img(800, 800*2.0/3.0, fname)


def make_rgb(mbobs):
    import images

    # SCALE=.015*np.sqrt(2.0)
    SCALE = 0.01
    # lsst
    # SCALE=0.0005
    # relative_scales = np.array([1.00, 1.2, 2.0])
    relative_scales = np.array([1.00, 1.0, 2.0])

    scales = SCALE*relative_scales

    r = mbobs[2][0].image
    g = mbobs[1][0].image
    b = mbobs[0][0].image

    rgb = images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=0.12,
    )
    return rgb


def compare_images_mosaic(im1, im2, **keys):
    import biggles
    import copy
    import images

    color1 = keys.get('color1', 'blue')
    color2 = keys.get('color2', 'orange')
    colordiff = keys.get('colordiff', 'red')

    label1 = keys.get('label1', 'im1')
    label2 = keys.get('label2', 'im2')

    cen = keys.get('cen', None)
    if cen is None:
        cen = [(im1.shape[0]-1)/2., (im1.shape[1]-1)/2.]

    labelres = '%s-%s' % (label1, label2)

    biggles.configure('default', 'fontsize_min', 1.)

    if im1.shape != im2.shape:
        raise ValueError("images must be the same shape")

    resid = im1-im2

    maxval = max(im1.max(), resid.max())
    mosaic = np.zeros((im1.shape[0], 3*im1.shape[1]))
    ncols = im1.shape[1]
    mosaic[:, 0:ncols] = im1/maxval
    mosaic[:, ncols:2*ncols] = im2/im2.max()
    mosaic[:, 2*ncols:3*ncols] = resid/maxval

    # will only be used if type is contour
    tab = biggles.Table(2, 1)
    if 'title' in keys:
        tab.title = keys['title']

    tkeys = copy.deepcopy(keys)
    tkeys.pop('title', None)
    tkeys['show'] = False
    tkeys['file'] = None
    tkeys['nonlinear'] = 0.4

    residplt = images.view(mosaic, **tkeys)

    dof = resid.size
    chi2per = (resid**2).sum()/dof
    lab = biggles.PlotLabel(0.9, 0.9,
                            r'$\chi^2/npix$: %.3e' % chi2per,
                            color='red',
                            halign='right')
    residplt.add(lab)

    cen0 = int(cen[0])
    cen1 = int(cen[1])
    im1rows = im1[:, cen1]
    im1cols = im1[cen0, :]
    im2rows = im2[:, cen1]
    im2cols = im2[cen0, :]
    resrows = resid[:, cen1]
    rescols = resid[cen0, :]

    him1rows = biggles.Histogram(im1rows, color=color1)
    him1cols = biggles.Histogram(im1cols, color=color1)
    him2rows = biggles.Histogram(im2rows, color=color2)
    him2cols = biggles.Histogram(im2cols, color=color2)
    hresrows = biggles.Histogram(resrows, color=colordiff)
    hrescols = biggles.Histogram(rescols, color=colordiff)

    him1rows.label = label1
    him2rows.label = label2
    hresrows.label = labelres
    key = biggles.PlotKey(0.1, 0.9, [him1rows, him2rows, hresrows])

    rplt = biggles.FramedPlot()
    rplt.add(him1rows, him2rows, hresrows, key)
    rplt.xlabel = 'Center Rows'

    cplt = biggles.FramedPlot()
    cplt.add(him1cols, him2cols, hrescols)
    cplt.xlabel = 'Center Columns'

    rplt.aspect_ratio = 1
    cplt.aspect_ratio = 1

    ctab = biggles.Table(1, 2)
    ctab[0, 0] = rplt
    ctab[0, 1] = cplt

    tab[0, 0] = residplt
    tab[1, 0] = ctab

    images._writefile_maybe(tab, **keys)
    images._show_maybe(tab, **keys)

    return tab


def compare_images_mosaic_old(im1, im2, **keys):
    import biggles
    import copy
    import images

    color1 = keys.get('color1', 'blue')
    color2 = keys.get('color2', 'orange')
    colordiff = keys.get('colordiff', 'red')

    label1 = keys.get('label1', 'im1')
    label2 = keys.get('label2', 'im2')

    cen = keys.get('cen', None)
    if cen is None:
        cen = [(im1.shape[0]-1)/2., (im1.shape[1]-1)/2.]

    labelres = '%s-%s' % (label1, label2)

    biggles.configure('default', 'fontsize_min', 1.)

    if im1.shape != im2.shape:
        raise ValueError("images must be the same shape")

    resid = im1-im2
    sresid = np.log10(resid - resid.min() + 1.0e-7)
    sresid -= sresid.min()
    sresid *= 1.0/sresid.max()

    logim1 = np.log10(im1.clip(min=1.0e-7))
    logim2 = np.log10(im2.clip(min=1.0e-7))

    logim1 -= logim1.min()
    logim2 -= logim2.min()

    maxval = max(logim1.max(), logim2.max())
    logim1 *= 1.0/maxval
    logim2 *= 1.0/maxval

    # will only be used if type is contour
    tab = biggles.Table(2, 1)
    if 'title' in keys:
        tab.title = keys['title']

    tkeys = copy.deepcopy(keys)
    tkeys.pop('title', None)
    tkeys['show'] = False
    tkeys['file'] = None

    mosaic = np.zeros((logim1.shape[0], 3*logim1.shape[1]))
    ncols = logim1.shape[1]
    mosaic[:, 0:ncols] = logim1
    mosaic[:, ncols:2*ncols] = logim2
    mosaic[:, 2*ncols:3*ncols] = sresid

    residplt = images.view(mosaic, **tkeys)

    dof = resid.size
    chi2per = (resid**2).sum()/dof
    lab = biggles.PlotLabel(0.9, 0.9,
                            r'$\chi^2/npix$: %.3e' % chi2per,
                            color='red',
                            halign='right')
    residplt.add(lab)

    cen0 = int(cen[0])
    cen1 = int(cen[1])
    im1rows = im1[:, cen1]
    im1cols = im1[cen0, :]
    im2rows = im2[:, cen1]
    im2cols = im2[cen0, :]
    resrows = resid[:, cen1]
    rescols = resid[cen0, :]

    him1rows = biggles.Histogram(im1rows, color=color1)
    him1cols = biggles.Histogram(im1cols, color=color1)
    him2rows = biggles.Histogram(im2rows, color=color2)
    him2cols = biggles.Histogram(im2cols, color=color2)
    hresrows = biggles.Histogram(resrows, color=colordiff)
    hrescols = biggles.Histogram(rescols, color=colordiff)

    him1rows.label = label1
    him2rows.label = label2
    hresrows.label = labelres
    key = biggles.PlotKey(0.1, 0.9, [him1rows, him2rows, hresrows])

    rplt = biggles.FramedPlot()
    rplt.add(him1rows, him2rows, hresrows, key)
    rplt.xlabel = 'Center Rows'

    cplt = biggles.FramedPlot()
    cplt.add(him1cols, him2cols, hrescols)
    cplt.xlabel = 'Center Columns'

    rplt.aspect_ratio = 1
    cplt.aspect_ratio = 1

    ctab = biggles.Table(1, 2)
    ctab[0, 0] = rplt
    ctab[0, 1] = cplt

    tab[0, 0] = residplt
    tab[1, 0] = ctab

    images._writefile_maybe(tab, **keys)
    images._show_maybe(tab, **keys)

    return tab


def compare_images(im1_in, im2_in, wt_in, **keys):
    import biggles
    import copy
    import images

    """
    wt = wt_in.copy()
    maxwt = wt.max()
    noiseval = np.sqrt(1.0/maxwt)

    w = np.where(wt <= 0.0)
    if w[0].size > 0:
        wt[w] = maxwt
    noise = np.random.normal(size=wt.shape)
    noise *= np.sqrt(1.0/wt)
    """

    if im1_in.shape != im2_in.shape:
        raise ValueError("images must be the same shape")

    color1 = keys.get('color1', 'blue')
    color2 = keys.get('color2', 'orange')
    colordiff = keys.get('colordiff', 'red')

    label1 = keys.get('label1', 'im1')
    label2 = keys.get('label2', 'im2')

    resid = (im1_in - im2_in)

    im1 = im1_in
    im2 = im2_in

    cen = [(im1.shape[0]-1)/2., (im1.shape[1]-1)/2.]

    labelres = '%s-%s' % (label1, label2)

    biggles.configure('default', 'fontsize_min', 1.)

    # will only be used if type is contour
    tab = biggles.Table(2, 3)
    if 'title' in keys:
        tab.title = keys['title']

    tkeys = copy.deepcopy(keys)
    tkeys.pop('title', None)
    tkeys['show'] = False
    tkeys['file'] = None

    autoscale = True
    tab[0, 0] = images.view(im1, autoscale=autoscale, **tkeys)
    tab[0, 1] = images.view(im2, autoscale=autoscale, **tkeys)
    tab[0, 2] = residplt = images.view(
        resid*np.sqrt(wt_in.clip(min=0)), **tkeys
    )

    wgood = np.where(wt_in > 0.0)
    dof = wgood[0].size
    chi2per = (resid**2 * wt_in).sum()/dof
    lab = biggles.PlotLabel(0.9, 0.9,
                            r'$\chi^2/dof$: %.2f' % chi2per,
                            color='red',
                            halign='right')
    residplt.add(lab)

    cen0 = int(cen[0])
    cen1 = int(cen[1])
    im1rows = im1_in[:, cen1]
    im1cols = im1_in[cen0, :]
    im2rows = im2_in[:, cen1]
    im2cols = im2_in[cen0, :]
    resrows = resid[:, cen1]
    rescols = resid[cen0, :]

    him1rows = biggles.Histogram(im1rows, color=color1)
    him1cols = biggles.Histogram(im1cols, color=color1)
    him2rows = biggles.Histogram(im2rows, color=color2)
    him2cols = biggles.Histogram(im2cols, color=color2)
    hresrows = biggles.Histogram(resrows, color=colordiff)
    hrescols = biggles.Histogram(rescols, color=colordiff)

    him1rows.label = label1
    him2rows.label = label2
    hresrows.label = labelres
    key = biggles.PlotKey(0.1, 0.9,
                          [him1rows, him2rows, hresrows])

    rplt = biggles.FramedPlot()
    rplt.add(him1rows, him2rows, hresrows, key)
    rplt.xlabel = 'Center Rows'

    cplt = biggles.FramedPlot()
    cplt.add(him1cols, him2cols, hrescols)
    cplt.xlabel = 'Center Columns'

    rplt.aspect_ratio = 1
    cplt.aspect_ratio = 1

    tab[1, 0] = rplt
    tab[1, 1] = cplt

    images._writefile_maybe(tab, **keys)
    images._show_maybe(tab, **keys)

    return tab
