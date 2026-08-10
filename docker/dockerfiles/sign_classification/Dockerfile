FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-opencv git libgl1-mesa-glx curl haveged && \
    rm -rf /var/lib/apt/lists/*

RUN echo "#!/bin/bash\nhaveged -w 1024 &\nexec \"\$@\"" > /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /app

COPY src /app/src
COPY services /app/services
COPY config /app/config
COPY requirements.txt /app/

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install -r requirements.txt

RUN mkdir -p /app/models/traffic_sign

ENV MODEL_PATH=/models/traffic_sign/traffic_sign_classification.h5
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/app/services/sign_classification_service.py"]
