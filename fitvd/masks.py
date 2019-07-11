from __future__ import print_function
from . import files


def load_mask(tilename=None):

    mask_fname = files.get_mask_file(tilename)
    bounds_fname = files.get_bounds_file(tilename)

    print('loading mask from:', mask_fname)
    print('loading bounds from:', bounds_fname)
    return Mask(mask_fname=mask_fname, bounds_fname=bounds_fname)


class Mask(object):
    def __init__(self, mask_fname, bounds_fname):
        self._mask_fname = mask_fname
        self._bounds_fname = bounds_fname
        self._load_masks()

    def _load_masks(self):
        import healsparse as hs
        self._mask_map = hs.HealSparseMap.read(self._mask_fname)
        self._bounds_map = hs.HealSparseMap.read(self._bounds_fname)

    def is_masked(self, ra, dec):
        """
        check if the input positions are masked
        """

        mask_values = self._mask_map.getValueRaDec(ra, dec)
        bounds_values = self._bounds_map.getValueRaDec(ra, dec)

        return (
            (mask_values > 0) | (bounds_values == 0)
        )

    def is_unmasked(self, ra, dec):
        """
        check if the input positions are masked
        """

        is_masked = self.is_masked(ra, dec)
        return ~is_masked
