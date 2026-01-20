FROM quay.io/fenicsproject/stable
USER root

RUN pip3 install -U --no-cache-dir --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org \
simplejson \
mpi4py \
h5py \
scikit-image \
tabulate \
meshpy \
scipy

RUN git clone https://github.com/mikaem/fenicstools.git && \
cd fenicstools && \
python3 setup.py install && \
cd ..
