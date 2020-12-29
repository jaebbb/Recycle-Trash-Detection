FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/jaebbb/Recycle-Trash-Detection.git /Recycle-Trash-Detection
WORKDIR /Recycle-Trash-Detection
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt

RUN pip install --upgrade pip
RUN python -V

RUN pip install mmpycocotools
RUN pip install mmcv-full==1.1.6+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html --use-deprecated=legacy-resolver
RUN pip install -v -e .