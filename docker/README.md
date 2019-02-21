# Creating a Docker/Shifter container


## Building the image

```sh
$ cd docker
$ docker build -t eiffl/fitvd:latest .
```

## Pushing the image to docker hub

```sh
$ docker push eiffl/fitvd:latest
```


## Downloading the image at NERSC

```sh
$ shifterimg pull eiffl/fitvd:latest
```

## Running fitvd inside shifter at NERSC

```sh
$ salloc -N 1 -q interactive -C haswell -t03:00:00 -L SCRATCH --image=eiffl/fitvd:latest
$ shifter fitvd-make-fofs \
    --conf=$config \
    --plot=fofs.png \
    --output=fofs.fits \
    SN-C3_C28_r3688p01/SN-C3_C28_r3688p01_r_meds-Y3A2_DEEP.fits.fz
```
