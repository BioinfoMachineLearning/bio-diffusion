FROM nvcr.io/nvidia/pytorch:21.06-py3

####################################################################

# install system requirements
RUN apt update && apt install -y vim libgl-dev libgl1
RUN apt-get install -y --reinstall ca-certificates  # note: for git
RUN apt install -y git

# set environment variables
ENV MPLCONFIGDIR /data/MPL_Config
ENV TORCH_HOME /data/Torch_Home
ENV TORCH_EXTENSIONS_DIR /data/Torch_Extensions

####################################################################

# set the entrypoint
CMD /bin/bash