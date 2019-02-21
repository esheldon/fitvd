# Creating a Docker/Shifter container

The container should be built automatically and kept updated with the master branch
so that in principle you don't have to build it yourself.


## Running fitvd inside shifter at NERSC

Make sure to grab the latest version of the container by running:
```sh
$ shifterimg pull esheldon/fitvd:latest
```
This doesn't have to be done all the time, only when you want to update your
install of fitvd.

Then to use the code, you can start an interactive session and use shifter to
encapsulate the call to fitvd:
```sh
$ salloc -N 1 -q interactive -C haswell -t03:00:00 -L SCRATCH --image=esheldon/fitvd:latest
$ shifter fitvd-make-fofs \
    --conf=$config \
    --plot=fofs.png \
    --output=fofs.fits \
    SN-C3_C28_r3688p01/SN-C3_C28_r3688p01_r_meds-Y3A2_DEEP.fits.fz
```
Alternatively, this can be done in a slurm job, checkout NERSC shifter documentation
to see how.

## Building and pushing the container manually

In case you need to build the container manually:
```sh
$ cd docker
$ docker build -t esheldon/fitvd:latest .
```
Then, to push the container to dockerhub:
```sh
$ docker push esheldon/fitvd:latest
```
