FROM jupyter/scipy-notebook

USER root

RUN apt-get update --yes

RUN conda install -c conda-forge gfortran
RUN conda install -c conda-forge mpi4py
RUN conda install git pip
RUN pip install git+https://github.com/antolonappan/cmblens.git

