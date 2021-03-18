FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

COPY environment.devenv.yml .
RUN conda env create -f environment.devenv.yml
RUN apt update && apt install -y gcc



COPY . .
# Set the default docker build shell to run as the conda wrapped process
SHELL ["conda", "run", "-n", "blue_zero", "/bin/bash", "-c" ]

# Set your entrypoint to use the conda environment as well
#ENTRYPOINT ["conda", "run", "-n", "blue_zero", "python", "run.py"]

RUN python setup.py install
