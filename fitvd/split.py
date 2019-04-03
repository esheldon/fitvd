from __future__ import print_function
import numpy as np
import esutil as eu
from numba import njit

def get_splits(num_fofs, chunksize):
    """
    split FoF groups into chunks
    """

    nchunks = num_fofs//chunksize
    if num_fofs % chunksize != 0:
        nchunks += 1

    fof_splits = []
    for chunk in range(nchunks):
        start = chunk*chunksize
        end = start + chunksize - 1
        if end > num_fofs:
            end = num_fofs
        fof_splits.append([start,end])

    return fof_splits

def get_splits_variable(fofs, chunksize, threshold, fast=True):
    """
    split FoF groups into chunks, with large FoF groups
    in their own chunk

    parameters
    ----------
    fofs: array with fields
        The FoF struct
    chunksize: int
        Nominal chunksize.  FoF groups that exceed the threshold
        in size are broken out into their own chunk.  E.g. if
        FoF group 25 is large, and chunksize is 10, the chunks
        will look like this

            0-9
            10-19
            20-24
            25
            26-35
            36-45
            ...

    theshold: int
        groups with more than this many members are put into their
        own chunk.
    """

    h, rev = eu.stat.histogram(fofs['fofid'], binsize=1, rev=True)

    nfofs = np.unique(fofs['fofid']).size

    if fast:
        fof_splits = make_splits_struct(nfofs)
        _get_splits_variable_fast(chunksize, threshold, h, rev, fof_splits)
    else:

        fof_splits = []

        nfofs = len(h)
        start=0
        for fofind in range(len(h)):
            if rev[fofind] != rev[fofind+1]:
                ind = rev[ rev[fofind]:rev[fofind+1] ]
                fofsize = ind.size

                end = fofind

                if fofsize <= threshold:
                    # continue building this chunk
                    current_size = end-start+1
                    if current_size == chunksize or fofind == nfofs-1:
                        # we reached our chunksize, store the split
                        fof_splits.append( (start,end) )
                        start=end+1

                else:
                    # we will put this FoF group into its own chunk
                    if start==end:
                        # we just started a new chunk, so just put it into
                        # its own chunk
                        fof_splits.append( (start,end) )
                        start = end+1
                    else:
                        # we were in the middle of a chunk, so store the
                        # old chunk and the new one
                        fof_splits.append( (start,end-1) )
                        fof_splits.append( (end,end) )
                        start = end+1

    return fof_splits

@njit
def _get_splits_variable_fast(chunksize, threshold, h, rev, fof_splits):

    nfofs = h.size

    start=0
    isplit=0
    for fofind in range(nfofs):
        if rev[fofind] != rev[fofind+1]:
            ind = rev[ rev[fofind]:rev[fofind+1] ]
            fofsize = ind.size

            end = fofind

            if fofsize <= threshold:
                # continue building this chunk
                current_size = end-start+1
                if current_size == chunksize or fofind == nfofs-1:
                    # we reached our chunksize, store the split
                    fof_splits['start'][isplit] = start
                    fof_splits['end'][isplit] = end
                    start=end+1
                    isplit += 1

            else:
                # we will put this FoF group into its own chunk
                if start==end:
                    # we just started a new chunk, so just put it into
                    # its own chunk
                    fof_splits['start'][isplit] = start
                    fof_splits['end'][isplit] = end
                    start=end+1
                    isplit += 1

                else:
                    # we were in the middle of a chunk, so store the
                    # old chunk and the new one

                    fof_splits['start'][isplit] = start
                    fof_splits['end'][isplit] = end-1
                    isplit += 1

                    fof_splits['start'][isplit] = end
                    fof_splits['end'][isplit] = end
                    start = end+1
                    isplit += 1

    return isplit

def get_splits_variable_fixnum(fofs, nsplits, threshold):
    """
    split FoF groups into chunks, with large FoF groups
    in their own chunk

    parameters
    ----------
    fofs: array with fields
        The FoF struct
    nsplits: int
        The number of splits to produce
    theshold: int
        groups with more than this many members are put into their
        own chunk.
    """

    h, rev = eu.stat.histogram(fofs['fofid'], binsize=1, rev=True)


    nfofs = np.unique(fofs['fofid']).size
    fof_splits = make_splits_struct(nfofs)

    # starting chunksize. We will need to increase it
    chunksize = nfofs//nsplits
    if chunksize < 1:
        chunksize = 1

    print('nfofs:',nfofs)
    print('target nsplits:',nsplits)

    while True:
        tnsplits = _get_splits_variable_fast(chunksize, threshold, h, rev, fof_splits)

        print('chunksize:',chunksize,'nsplits:',tnsplits)
        if tnsplits == nsplits:
            break

        diff = tnsplits - nsplits

        if diff < 0:
            # we crossed 0 so we didn't find it.
            # just lump the last bits together
            last_chunksize = chunksize-1
            tnsplits = _get_splits_variable_fast(last_chunksize, threshold, h, rev, fof_splits)
            fof_splits['end'][nsplits-1] = fof_splits['end'][tnsplits-1]

            break

        chunksize += 1

    return fof_splits[0:nsplits]

def make_splits_struct(num):
    """
    make a fof splits struct array
    """
    return np.zeros(num, dtype=[('start','i8'),('end','i8')])
