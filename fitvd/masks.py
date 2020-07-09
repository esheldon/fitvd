

def load_mask(*, mask_file=None, bounds_file=None):
    """
    load region masks
    """
    import desmasks

    if mask_file is not None or bounds_file is not None:
        assert bounds_file is not None and mask_file is not None, \
            'send both mask and bounds'

        print('loading mask:', mask_file)
        print('loading bounds:', bounds_file)
        mask = desmasks.TileMask(
            mask_fname=mask_file,
            bounds_fname=bounds_file,
        )
    else:
        mask = None

    return mask


def load_objmask(*, mask_file=None):
    """
    load an object-level mask
    """
    import desmasks

    if mask_file is not None:
        print('loading objmask:', mask_file)
        objmask = desmasks.ObjMask(mask_file)
    else:
        objmask = None

    return objmask
