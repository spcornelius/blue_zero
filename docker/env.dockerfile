# Dockerfile for mmdanziger/blue_zero
# This container has the environment for blue_zero but no code.
# To get a version with code either run with the GH_TOKEN flag
# and a valid GH_TOKEN in your env, or use it as a base image
# as shown in dev.dockerfile
FROM nvidia/cuda:11.0-base-ubuntu20.04
LABEL org.opencontainers.image.authors="mmdanziger@gmail.com"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.11 \
 && conda clean -ya

COPY environment.devenv.yml .
RUN conda env create -f environment.devenv.yml
ENV PATH /home/user/miniconda/envs/blue_zero/bin/:$PATH
RUN conda run -n blue_zero pip install jupyterlab

# Set your entrypoint to use the conda environment as well
#ENTRYPOINT ["conda", "run", "-n", "blue_zero", "python", "run.py"]
ADD ./docker/blue_zero_entrypoint.sh entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD ["bash"]
