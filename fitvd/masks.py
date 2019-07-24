import desmasks


def load_mask(*, mask_file=None, bounds_file=None):
    """
    load region masks
    """
    if mask_file is not None or bounds_file is not None:
        assert bounds_file is not None and mask_file is not None, \
            'send both mask and bounds'

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
    if mask_file is not None:
        objmask = desmasks.ObjMask(mask_file)
    else:
        objmask = None

    return objmask
