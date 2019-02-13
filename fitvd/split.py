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
