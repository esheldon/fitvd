"""
TODO

    - condor submit with master script rather than individual scripts?
    - wq probably makes more sense but it is totally booked up
    - batch maker just takes meds run and conf and figures
    out the rest?
    - make it write a fof making script
      note it uses multiple cores via numba+mkl
"""
from __future__ import print_function

import os
import numpy as np
import yaml
import logging
import fitsio
import meds
import desmasks
from . import split
from . import files
from . import fofs

logger = logging.getLogger(__name__)


class ShellCollateBatch(dict):
    def __init__(self, args):
        self.args = args
        self._load_configs()

        self._make_dirs()

    def go(self):
        """
        write scripts for all tiles
        """
        for tilename in self.tile_conf['tilenames']:
            self.go_tile(tilename)

    def go_tile(self, tilename):
        """
        write the script to collate the files
        """

        # if 'fofs' not in self.fit_conf:
        #     meds_text = '--meds=%s' % self.meds_info[tilename][0]
        # else:
        #     meds_text = ''
        meds_text = ''

        text = _collate_script_template % {
            'run': self['run'],
            'fit_config': self['fit_config'],
            'n': self.args.n,
            'tilename': tilename,
            'meds_text': meds_text,
        }

        collate_script = files.get_collate_script_path(self['run'], tilename)

        print('writing script:', collate_script)
        with open(collate_script, 'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % collate_script)

    def _make_dirs(self):
        dirs = [
            files.get_script_dir(self['run']),
            files.get_collated_dir(self['run']),
        ]
        for d in dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass

    def _load_configs(self):

        with open(self.args.run_config) as fobj:
            run_config = yaml.load(fobj)
        self.update(run_config)

        self['fit_config'] = os.path.abspath(
            os.path.expandvars(
                self['fit_config']
            )
        )
        with open(self['fit_config']) as fobj:
            self.fit_conf = yaml.load(fobj)

        bname = os.path.basename(self.args.run_config)
        self['run'] = bname.replace('.yaml', '')

        self.tile_conf = files.read_yaml(self.args.tile_config)
        self.meds_info = _get_meds_file_info(self, self.tile_conf)


class WQCollateBatch(ShellCollateBatch):
    def go_tile(self, tilename):
        """
        write WQ scripts
        """

        job_name = 'collate-%s-%s' % (self['run'], tilename)

        # if 'fofs' not in self.fit_conf:
        #     meds_text = '--meds=%s' % self.meds_info[tilename][0]
        # else:
        #     meds_text = ''
        meds_text = ''

        text = _collate_wq_template % {
            'run': self['run'],
            'fit_config': self['fit_config'],
            'job_name': job_name,
            'n': self.args.n,
            'conda_env': self.args.conda_env,
            'tilename': tilename,
            'meds_text': meds_text,
        }
        wq_script = files.get_wq_collate_script_path(self['run'], tilename)

        wqlog = wq_script + '.wqlog'
        if os.path.exists(wqlog):
            try:
                os.remove(wqlog)
            except FileNotFoundError:
                pass

        print('writing:', wq_script)
        with open(wq_script, 'w') as fobj:
            fobj.write(text)


class FoFBatchBase(dict):
    def __init__(self, args):
        self.args = args

        self.run_conf = files.read_yaml(self.args.run_config)
        self.tile_conf = files.read_yaml(self.args.tile_config)

        self['fit_config'] = os.path.abspath(
            os.path.expandvars(
                self.run_conf['fit_config']
            )
        )
        self.fit_conf = files.read_yaml(self['fit_config'])
        # assert 'fofs' in self.fit_conf, 'fofs entry must be in fit config'

        self['run'] = files.extract_run_from_config(self.args.run_config)
        self.meds_info = _get_meds_file_info(self.run_conf, self.tile_conf)

        self._make_dirs()

    def go(self):
        """
        write batch files for all tiles
        """
        for tilename in self.tile_conf['tilenames']:
            self.go_tile(tilename)

    def go_tile(self, tilename):
        """
        write the script to make the fof groups
        """
        fof_file = files.get_fof_file(self['run'], tilename)
        plot_file = fof_file.replace('.fits', '.png')

        meds_files = self.meds_info[tilename]

        fof_band = self.run_conf['fof_band']

        if self.fit_conf.get('use_mask', False):
            uvista_bands = self.run_conf.get('uvista_bands', [])
            video_bands = self.run_conf.get('video_bands', [])

            if len(uvista_bands) != 0:
                with_uvista = True
            else:
                with_uvista = False


            mask_file = desmasks.files.get_mask_file(
                tilename,
                with_uvista=with_uvista,
            )
            bounds_file = desmasks.files.get_bounds_file(tilename)
            mask_text = '--mask=%s --bounds=%s' % (mask_file, bounds_file)

            if len(video_bands) != 0:
                objmask_file = _get_objmask_file(
                    self.run_conf,
                    self.tile_conf,
                    tilename,
                )
                mask_text = '%s --objmask=%s' % (mask_text, objmask_file)
        else:
            mask_text = ''

        text = _fof_script_template % {
            'fof_file': fof_file,
            'plot_file': plot_file,
            'fit_config': self['fit_config'],
            'meds_file': meds_files[fof_band],
            'mask_text': mask_text,
        }

        fof_script = files.get_fof_script_path(self['run'], tilename)
        print('writing fof script:', fof_script)
        with open(fof_script, 'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % fof_script)

    def _make_dirs(self):
        dirs = [
            files.get_script_dir(self['run']),
            files.get_fof_dir(self['run']),
        ]
        for d in dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass


class ShellFoFBatch(FoFBatchBase):
    pass


class WQFoFBatch(FoFBatchBase):
    def go_tile(self, tilename):
        """
        write WQ scripts
        """
        # this will write the basic script
        super(WQFoFBatch, self).go_tile(tilename)

        # now write the submit script
        fof_script = files.get_fof_script_path(self['run'], tilename)

        job_name = '%s-%s-make-fofs' % (self['run'], tilename)

        text = _wq_template % {
            'script': fof_script,
            'job_name': job_name,
            'conda_env': self.args.conda_env,
        }
        wq_script = files.get_wq_fof_script_path(self['run'], tilename)

        wqlog = wq_script + '.wqlog'
        if os.path.exists(wqlog):
            try:
                os.remove(wqlog)
            except FileNotFoundError:
                pass

        print('writing:', wq_script)
        with open(wq_script, 'w') as fobj:
            fobj.write(text)


class ShellBatch(dict):
    def __init__(self, args):
        self.args = args
        self._load_configs()

        self['fit_config'] = os.path.abspath(
            os.path.expandvars(
                self['fit_config']
            )
        )
        self.fit_conf = files.read_yaml(self['fit_config'])

        self._set_rng()
        self._make_dirs()

    def go(self):
        """
        write scripts for all tiles
        """
        for tilename in self.tile_conf['tilenames']:
            self.rng.seed(self.tile_seeds[tilename])
            self.go_tile(tilename)

    def go_tile(self, tilename):
        """
        write all FoF groups in the tile
        """

        fofst = self._get_fofs(tilename)
        fof_splits = split.get_splits_variable(
            fofst,
            self['chunksize'],
            self['threshold'],
        )

        for isplit, fof_split in enumerate(fof_splits):

            start, end = fof_split
            if self.args.skip_large and start == end:
                logger.info('skipping large: %s' % start)
                continue

            logger.info('%s %s' % (isplit, fof_split))
            self._write_split(tilename, isplit, fof_split)

    def _make_dirs(self):
        dirs = [
            files.get_collated_dir(self['run']),
        ]

        for tilename in self.meds_info:
            dirs += [
                files.get_split_script_dir(self['run'], tilename),
                files.get_split_dir(self['run'], tilename),
            ]

        for d in dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass

    def _write_split(self, tilename, isplit, fof_split):
        """
        just write out the scripts, no submit files
        """
        self._write_script(tilename, isplit, fof_split)

    def _write_script(self, tilename, isplit, fof_split):
        start, end = fof_split
        fname = files.get_split_script_path(self['run'], tilename, start, end)

        output_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='fits',
        )
        log_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='log',
        )

        if self.args.missing and os.path.exists(output_file):
            if os.path.exists(fname):
                os.remove(fname)
            return

        meds_files = self.meds_info[tilename]
        meds_files = ' '.join(meds_files)

        d = {}
        d['seed'] = self._get_seed()
        d['output_file'] = os.path.abspath(output_file)
        d['fit_config'] = self['fit_config']

        d['start'] = start
        d['end'] = end
        d['meds_files'] = meds_files
        d['logfile'] = os.path.abspath(log_file)

        fof_file = files.get_fof_file(self['run'], tilename)
        d['fofs_text'] = '--fofs=%s' % fof_file
        """
        if 'fofs' in self.fit_conf:
            fof_file = files.get_fof_file(self['run'], tilename)
            d['fofs_text'] = '--fofs=%s' % fof_file
        else:
            d['fofs_text'] = ''
        """

        if 'model_pars_run' in self:
            pars_file = files.get_collated_file(
                self['model_pars_run'],
                tilename,
            )
            d['model_pars'] = '--model-pars=%s' % pars_file
        else:
            d['model_pars'] = ''

        if self.args.offsets is not None:
            d['offsets'] = '--offsets=%s' % self.args.offsets
        else:
            d['offsets'] = ''

        if self.args.blacklist is not None:
            d['blacklist'] = '--blacklist=%s' % self.args.blacklist
        else:
            d['blacklist'] = ''

        text = _script_template % d

        logger.info('script: %s' % fname)
        with open(fname, 'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % fname)

    def _get_seed(self):
        return self.rng.randint(0, 2**31)

    def _load_configs(self):
        with open(self.args.run_config) as fobj:
            run_config = yaml.load(fobj)
        self.update(run_config)

        bname = os.path.basename(self.args.run_config)
        self['run'] = bname.replace('.yaml', '')

        self.tile_conf = files.read_yaml(self.args.tile_config)
        self.meds_info = _get_meds_file_info(self, self.tile_conf)

    def _get_fofs(self, tilename):
        fof_file = files.get_fof_file(self['run'], tilename)
        nbrs, fofst = files.load_fofs(fof_file)
        """
        if 'fofs' in self.fit_conf:
            fof_file = files.get_fof_file(self['run'], tilename)
            nbrs, fofst = files.load_fofs(fof_file)
        else:
            meds_file = self.meds_info[tilename][0]
            m = meds.MEDS(meds_file)
            cat = m.get_cat()
            fofst = fofs.make_singleton_fofs(cat)
        """

        return fofst

    def _set_rng(self):
        self.rng = np.random.RandomState(self['seed'])
        self.tile_seeds = {}
        for tilename in self.meds_info:
            self.tile_seeds[tilename] = self.rng.randint(low=0, high=2**15)


class WQBatch(ShellBatch):
    """
    just write out the scripts, no submit files
    """
    def _write_split(self, tilename, isplit, fof_split):
        super(WQBatch, self)._write_split(tilename, isplit, fof_split)
        self._write_wq_script(tilename, isplit, fof_split)

    def _write_wq_script(self, tilename, isplit, fof_split):
        """
        write the wq submit script
        """
        args = self.args
        start, end = fof_split

        script_file = files.get_split_script_path(
            self['run'],
            tilename,
            start,
            end,
        )

        wq_file = files.get_split_wq_path(self['run'], tilename, start, end)
        job_name = '%s-%s-%06d-%06d' % (self['run'], tilename, start, end)

        output_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='fits',
        )

        if args.missing or args.verify:
            file_exists = os.path.exists(output_file)
            if not file_exists:
                ok = False
            else:
                if args.verify:
                    with fitsio.FITS(output_file) as fits:
                        if ('model_fits' in fits and 'epochs_data' in fits):
                            ok = True
                        else:
                            print('extensions missing')
                            ok = False
                else:
                    ok = True

            if ok:
                if os.path.exists(wq_file):
                    os.remove(wq_file)
                return

        logger.info('wq script: %s' % wq_file)

        d = {}
        d['script'] = script_file
        d['job_name'] = job_name
        d['conda_env'] = args.conda_env

        text = _wq_template % d

        wqlog = wq_file + '.wqlog'
        if os.path.exists(wqlog):
            try:
                os.remove(wqlog)
            except FileNotFoundError:
                pass

        with open(wq_file, 'w') as fobj:
            fobj.write(text)


class CondorBatch(ShellBatch):
    """
    just write out the scripts, no submit files
    """

    def go(self):
        """
        write all the scripts
        """

        super(CondorBatch, self).go()

    def go_tile(self, tilename):
        """
        write condor files for one tile
        """

        assert not self.args.verify, 'no verify for condor yet'

        self._clean_condor_files(tilename)

        self._write_master(tilename)

        fofst = self._get_fofs(tilename)
        fof_splits = split.get_splits_variable(
            fofst,
            self['chunksize'],
            self['threshold'],
        )

        njobs = 0
        fobj = None

        icondor = 0
        for isplit, fof_split in enumerate(fof_splits):
            start, end = fof_split
            if self.args.skip_large and start == end:
                continue

            if (self.args.missing and
                    self._split_exists(tilename, isplit, fof_split)):
                continue

            if njobs % self['jobs_per_sub'] == 0:
                if fobj is not None:
                    fobj.close()
                fobj = self._open_condor_script(tilename, icondor)
                icondor += 1

            self._write_split(fobj, tilename, isplit, fof_split)

            njobs += 1

    def _split_exists(self, tilename, isplit, fof_split):
        start, end = fof_split

        output_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='fits',
        )
        return os.path.exists(output_file)

    def _write_split(self, fobj, tilename, isplit, fof_split):
        """
        write the lines to the submit file object
        """

        start, end = fof_split

        output_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='fits',
        )
        log_file = files.get_split_output(
            self['run'],
            tilename,
            start,
            end,
            ext='log',
        )

        d = {}
        d['seed'] = self._get_seed()
        d['output_file'] = os.path.abspath(output_file)
        d['fit_config'] = self['fit_config']
        d['start'] = start
        d['end'] = end
        d['logfile'] = os.path.abspath(log_file)
        d['job_name'] = '%s-%s-%06d-%06d' % (self['run'], tilename, start, end)

        job = _condor_job_template % d

        fobj.write(job)

    def _make_dirs(self):
        dirs = [
            files.get_collated_dir(self['run']),
        ]

        for tilename in self.meds_info:
            dirs += [
                files.get_split_dir(self['run'], tilename),
                files.get_condor_dir(self['run'], tilename),
            ]

        for d in dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass

    def _write_master(self, tilename):
        """
        write the master script
        """
        meds_files = self.meds_info[tilename]
        meds_files = ' '.join(meds_files)

        fof_file = files.get_fof_file(self['run'], tilename)
        fofs_text = '--fofs=%s' % fof_file
        """
        if 'fofs' in self.fit_conf:
            fof_file = files.get_fof_file(self['run'], tilename)
            fofs_text = '--fofs=%s' % fof_file
        else:
            fofs_text = ''
        """
        d = {
            'meds_files': meds_files,
            'fofs_text': fofs_text,
        }

        if 'model_pars_run' in self:
            pars_file = files.get_collated_file(
                self['model_pars_run'],
                tilename,
            )
            d['model_pars'] = '--model-pars=%s' % pars_file
        else:
            d['model_pars'] = ''

        if self.args.offsets is not None:
            d['offsets'] = '--offsets=%s' % self.args.offsets
        else:
            d['offsets'] = ''

        if self.args.blacklist is not None:
            d['blacklist'] = '--blacklist=%s' % self.args.blacklist
        else:
            d['blacklist'] = ''


        text = _condor_master_template % d
        master_script = files.get_condor_master_path(self['run'], tilename)
        print('writing master:', master_script)
        with open(master_script, 'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % master_script)

    def _clean_condor_files(self, tilename):
        for icondor in range(1000):
            cname = files.get_condor_script(self['run'], tilename, icondor)
            sname = cname+'.submitted'
            for fname in [cname, sname]:
                if os.path.exists(fname):
                    try:
                        os.remove(fname)
                    except FileNotFoundError:
                        pass

    def _open_condor_script(self, tilename, icondor):
        """
        open the condor script
        """

        fname = files.get_condor_script(self['run'], tilename, icondor)
        print('condor script:', fname)
        fobj = open(fname, 'w')

        master_script = files.get_condor_master_path(self['run'], tilename)
        text = _condor_head % {
            'master_script': master_script,
        }
        fobj.write(text)

        return fobj


_collate_script_template = r"""#!/bin/bash

run="%(run)s"

mpirun -n %(n)d fitvd-collate-mpi \
    %(meds_text)s \
    --run-config=$FITVD_CONFIG_DIR/${run}.yaml \
    --fit-config=%(fit_config)s \
    --tilename=%(tilename)s
"""

_collate_wq_template = r"""
command: |
    . ~/.bashrc
    source activate %(conda_env)s

    mpirun -hostfile %%hostfile%% fitvd-collate-mpi \
        %(meds_text)s \
        --run-config=$FITVD_CONFIG_DIR/%(run)s.yaml \
        --fit-config=%(fit_config)s \
        --tilename=%(tilename)s


job_name: %(job_name)s
N: %(n)d
hostfile: auto
"""

_fof_script_template = r"""#!/bin/bash

fof_file="%(fof_file)s"
plot_file="%(plot_file)s"
config_file="%(fit_config)s"
meds_file="%(meds_file)s"

fitvd-make-fofs \
    --conf=$config_file \
    --plot=$plot_file \
    --output=$fof_file \
    %(mask_text)s \
    $meds_file

"""

_script_template = r"""#!/bin/bash

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

export OMP_NUM_THREADS=1



seed="%(seed)s"
output="%(output_file)s"
config="%(fit_config)s"
start="%(start)d"
end="%(end)d"
meds="%(meds_files)s"
logfile="%(logfile)s"

logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase

fitvd \
    --seed=$seed \
    --config=$config \
    --output=$output \
    --start=$start \
    --end=$end \
    %(fofs_text)s \
    %(model_pars)s \
    %(offsets)s \
    %(blacklist)s \
    $meds &> $tmplog

mv -vf $tmplog $logfile
"""


_wq_template = r"""
command: |
    . ~/.bashrc
    source activate %(conda_env)s
    bash %(script)s

job_name: %(job_name)s
"""

_condor_head = r"""
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable      = %(master_script)s

Image_Size       =  1000000

GetEnv = True

kill_sig        = SIGINT

#requirements = (cpu_experiment == "sdcc")

+Experiment     = "astro"
"""

_condor_job_template = """
+job_name = "%(job_name)s"
Arguments = %(seed)d %(output_file)s %(fit_config)s %(start)d %(end)d %(logfile)s

Queue
"""  # noqa


_condor_master_template = r"""#!/bin/bash

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

export OMP_NUM_THREADS=1

seed="$1"
output="$2"
config="$3"
start="$4"
end="$5"
logfile="$6"

logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase

meds="%(meds_files)s"

fitvd \
    --seed=$seed \
    --config=$config \
    --output=$output \
    --start=$start \
    --end=$end \
    %(model_pars)s \
    %(offsets)s \
    %(fofs_text)s \
    %(blacklist)s \
    $meds &> $tmplog

mv -vf $tmplog $logfile
"""


def _get_meds_file_info(run_conf, tile_conf):
    """
    returns a dict keyed by tilename, holding a list of
    meds files for each
    """
    fi = {}

    des_bands = run_conf.get('des_bands', [])
    video_bands = run_conf.get('video_bands', [])
    uvista_bands = run_conf.get('uvista_bands', [])

    for tilename_full in tile_conf['tilenames']:
        if 'DES' in tilename_full:
            tilename = tilename_full.split('_')[0]
        else:
            tilename = tilename_full

        meds_list = []

        for band in des_bands:
            fname = tile_conf['des_pattern'] % dict(
                tilename=tilename,
                tilename_full=tilename_full,
                band=band,
            )
            meds_list.append(fname)

        for band in video_bands:
            fname = tile_conf['video_pattern'] % dict(
                tilename=tilename,
                tilename_full=tilename_full,
                band=band,
            )
            meds_list.append(fname)

        for band in uvista_bands:
            fname = tile_conf['uvista_pattern'] % dict(
                tilename=tilename,
                tilename_full=tilename_full,
                band=band,
            )
            meds_list.append(fname)

        fi[tilename_full] = meds_list

    return fi


def _get_objmask_file(run_conf, tile_conf, tilename_full):
    """
    returns a dict keyed by tilename, holding a list of
    meds files for each
    """
    fi = {}

    video_bands = run_conf.get('video_bands', [])
    assert len(video_bands) != 0, 'need video bands to get objmask'

    if 'DES' in tilename_full:
        tilename = tilename_full.split('_')[0]
    else:
        tilename = tilename_full

    meds_fname = tile_conf['des_pattern'] % dict(
        tilename=tilename,
        tilename_full=tilename_full,
        band='J',
    )

    d = os.path.dirname(meds_fname)
    fname = '%s_comb_meds-VIDEO_DEEPmaskedobj.txt' % tilename_full

    return os.path.join(d, fname)
