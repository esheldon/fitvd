FROM conda/miniconda3:latest
MAINTAINER EiffL <francois.lanusse@gmail.com>

# Installs additional software dependencies
RUN apt-get update  && apt-get install -y libplot-dev plotutils git build-essential

# Installs conda dependencies
RUN conda update -y conda && conda install -y numba=0.40 pyyaml scipy fitsio

# Installs all of Erin's stack
RUN pip install git+https://github.com/esheldon/esutil \
    && pip install git+https://github.com/esheldon/meds@v0.9.6 \
    && pip install git+https://github.com/esheldon/ngmix \
    && pip install git+https://github.com/esheldon/psfex \
    && pip install git+https://github.com/esheldon/mof

# Adds optional dependencies
RUN pip install biggles \
    && pip install git+https://github.com/esheldon/images

# Install fitvd
RUN pip install git+https://github.com/esheldon/fitvd
