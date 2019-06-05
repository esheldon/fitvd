from __future__ import print_function

import os
import shutil
import numpy as np
import logging
import fitsio
import yaml
import ngmix

logger = logging.getLogger(__name__)

def read_yaml(fname):
    """
    read the yaml file
    """
    with open(fname) as fobj:
        data = yaml.load(fobj)

    return data

def extract_run_from_config(conf_name):
    """
    extrac the run id as the part before .yaml
    """
    bname=os.path.basename(conf_name)
    return bname.replace('.yaml','')


def get_fitvd_dir():
    """
    base dir, set in FITVD_DIR
    """
    return os.environ['FITVD_DIR']

def get_mask_dir():
    """
    get the collated file name
    """
    bdir = get_fitvd_dir()
    return os.path.join(
        bdir,
        'masks',
    )

def get_mask_file(tilename):
    """
    get the collated file name

    Parameters
    ----------
    tilename: string
        Either the basic tilename such as SN-C3_C10
        or with reqnum/attnum SN-C3_C10_r3688p01
    """
    d = get_mask_dir()

    if '_r' in tilename:
        tilename = '_'.join( tilename.split('_')[0:2] )

    return os.path.join(
        d,
        'mask-%s.dat' % tilename
    )


#
# directories
#

def get_tempdir():
    """
    temporary dir must be defined
    """
    return os.environ['TMPDIR']

def get_run_dir(run):
    """
    get the base run dir
    """
    bdir = get_fitvd_dir()
    return os.path.join(
        bdir,
        run,
    )

def get_fof_dir(run):
    """
    get the directory holding fofs
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'fofs',
    )

def get_split_dir(run, tilename):
    """
    get the split output directory
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'splits',
        tilename,
    )

def get_script_dir(run):
    """
    directory for scripts
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'scripts',
    )

def get_collated_dir(run):
    """
    get the collated directory
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'collated',
    )



def get_fof_file(run, tilename):
    """
    get the directory holding fofs
    """
    fof_dir=get_fof_dir(run)
    fname='%s-%s-fofs.fits' % (run, tilename)
    return os.path.join(
        fof_dir,
        fname,
    )


def get_collated_file(run, tilename):
    """
    get the collated file name
    """
    split_dir=get_collated_dir(run)
    fname = '%s-%s.fits' % (run, tilename)
    return os.path.join(
        split_dir,
        fname,
    )

def get_split_output(run, tilename, start, end, ext='fits'):
    """
    get the split output file
    """
    split_dir=get_split_dir(run, tilename)
    fname = '%s-%s-%06d-%06d.%s' % (run, tilename, start, end, ext)
    return os.path.join(
        split_dir,
        fname,
    )

def get_split_script_dir(run, tilename):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)
    return os.path.join(
        script_dir,
        tilename,
    )


def get_split_script_path(run, tilename, start, end):
    """
    chunk processing script
    """
    script_dir=get_split_script_dir(run, tilename)

    fname = '%s-%s-%06d-%06d.sh' % (run, tilename, start, end)
    return os.path.join(
        script_dir,
        fname,
    )

def get_collate_script_path(run, tilename):
    """
    script to run the collation
    """
    script_dir=get_script_dir(run)

    fname = '%s-collate-%s.sh' % (run, tilename)
    return os.path.join(
        script_dir,
        fname,
    )

def get_wq_collate_script_path(run, tilename):
    """
    script to run the collation
    """
    script = get_collate_script_path(run, tilename)
    return script.replace('.sh','.yaml')


def get_fof_script_path(run, tilename):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)

    fname = '%s-%s-make-fofs.sh' % (run, tilename)
    return os.path.join(
        script_dir,
        fname,
    )

def get_wq_fof_script_path(run, tilename):
    """
    directory for scripts
    """
    script = get_fof_script_path(run, tilename)
    return script.replace('.sh','.yaml')



def get_split_wq_path(run, tilename, start, end):
    """
    directory for scripts
    """
    script_dir=get_split_script_dir(run, tilename)

    fname = '%s-%s-%06d-%06d.yaml' % (run, tilename, start, end)
    return os.path.join(
        script_dir,
        fname,
    )


def get_condor_dir(run, tilename):
    """
    directory for scripts
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'condor',
        tilename,
    )

def get_condor_master_path(run, tilename):
    """
    master script for condor
    """
    condor_dir=get_condor_dir(run, tilename)

    fname = '%s-%s-master.sh' % (run, tilename)
    return os.path.join(
        condor_dir,
        fname,
    )

def get_condor_script(run, tilename, icondor):
    """
    submit script
    """
    condor_dir=get_condor_dir(run, tilename)

    fname = '%s-%s-%06d.condor' % (run, tilename, icondor)
    return os.path.join(
        condor_dir,
        fname,
    )




def load_fofs(fof_filename):
    """
    load FoF information from the file
    """
    logger.info('loading fofs: %s' % fof_filename)
    with fitsio.FITS(fof_filename) as fits:
        nbrs=fits['nbrs'][:]
        fofs=fits['fofs'][:]

    return nbrs, fofs

class StagedOutFile(object):
    """
    A class to represent a staged file
    If tmpdir=None no staging is performed and the original file
    path is used
    parameters
    ----------
    fname: string
        Final destination path for file
    tmpdir: string, optional
        If not sent, or None, the final path is used and no staging
        is performed
    must_exist: bool, optional
        If True, the file to be staged must exist at the time of staging
        or an IOError is thrown. If False, this is silently ignored.
        Default False.
    examples
    --------

    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedOutFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path,'w') as fobj:
            fobj.write("some data")

    """
    def __init__(self, fname, tmpdir=None, must_exist=False):

        self.must_exist = must_exist
        self.was_staged_out = False

        self._set_paths(fname, tmpdir=tmpdir)


    def _set_paths(self, fname, tmpdir=None):
        fname=expandpath(fname)

        self.final_path = fname

        if tmpdir is not None:
            self.tmpdir = expandpath(tmpdir)
        else:
            self.tmpdir = tmpdir

        fdir = os.path.dirname(self.final_path)

        if self.tmpdir is None:
            self.is_temp = False
            self.path = self.final_path
        else:
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)

            bname = os.path.basename(fname)
            self.path = os.path.join(self.tmpdir, bname)

            if self.tmpdir==fdir:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp = False
            else:
                self.is_temp = True

    def stage_out(self):
        """
        if a tempdir was used, move the file to its final destination
        note you normally would not call this yourself, but rather use a
        context, in which case this method is called for you
        with StagedOutFile(fname,tmpdir=tmpdir) as sf:
            #do something
        """

        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    mess = "temporary file not found: %s" % self.path
                    raise IOError(mess)
                else:
                    return

            if os.path.exists(self.final_path):
                print("removing existing file:",self.final_path)
                os.remove(self.final_path)

            makedir_fromfile(self.final_path)

            print("staging out '%s' -> '%s'" % (self.path,self.final_path))
            shutil.move(self.path,self.final_path)

        self.was_staged_out=True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()

def expandpath(path):
    """
    expand environment variables, user home directories (~), and convert
    to an absolute path
    """
    path=os.path.expandvars(path)
    path=os.path.expanduser(path)
    path=os.path.realpath(path)
    return path


def makedir_fromfile(fname):
    """
    extract the directory and make it if it does not exist
    """
    dname=os.path.dirname(fname)
    try_makedir(dname)

def try_makedir(dir):
    """
    try to make the directory
    """
    if not os.path.exists(dir):
        try:
            print("making directory:",dir)
            os.makedirs(dir)
        except:
            # probably a race condition
            pass


def get_psfex_name(meds_filename):
    """
    get the DES psfex name given the meds filename
    """
    d,f = os.path.split(meds_filename)
    front = f[0: f.find('_meds') ]
    fname = '%s_%s' % (front, 'psfcat.psf')
    return os.path.join(d, fname)

class MEDSPSFEx(ngmix.medsreaders.NGMixMEDS):
    """
    meds reader for one epoch and using a psfex
    file for psfs
    """
    def __init__(self, meds_filename, psfex_name=None):
        self._load_psfex(meds_filename, psfex_name)
        super(MEDSPSFEx,self).__init__(meds_filename)

        assert np.all(self['ncutout']==1),'only support a single cutout'

    def has_psf(self):
        """
        returns True if psfs are in the file
        """
        return True

    def get_psf(self, iobj, icutout):
        """
        Get a single psf image for the indicated entry.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        psf : np.array
            The PSF as a numpy array.
        """

        row = self['orig_row'][iobj, icutout]
        col = self['orig_col'][iobj, icutout]
        return self.psfex.get_rec(row, col)

    def _load_psfex(self, meds_filename, psfex_name=None):
        """
        load the psfex object from the file
        """
        import psfex
        if psfex_name is None:
            psfex_name = get_psfex_name(meds_filename)

        logger.info('loading psfex: %s' % psfex_name)
        self.psfex = psfex.PSFEx(psfex_name)


def read_blacklist(fname):
    """
    read a black list file

    The format should be one string per line
    """

    blacklist ={}
    logger.info('loading blacklist: %s' % fname)
    with open(fname) as fobj:
        for line in fobj:
            name = line.strip()
            blacklist[name] = {'name': name}

    return blacklist

