FROM lsstsqre/centos:7-stack-lsst_distrib-v15_0
MAINTAINER EiffL <francois.lanusse@gmail.com>

# Activate the stack environment
RUN source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib \
    && conda install -y numba pyyaml scipy \
    && pip install fitsio \
    && pip install git+https://github.com/esheldon/esutil \
    && pip install git+https://github.com/esheldon/meds@psfex2 \
    && pip install git+https://github.com/esheldon/ngmix \
    && pip install git+https://github.com/esheldon/psfex \
    && pip install git+https://github.com/esheldon/mof

RUN source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib \
    && pip install git+https://github.com/esheldon/fitvd

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["bash", "/entrypoint.sh"]
