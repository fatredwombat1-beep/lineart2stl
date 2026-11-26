# Python 3.12: mediapipe 0.10.21 is the last release that bundles the
# face/segmentation models used by photo mode.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATA_DIR=/data KEEP_HOURS=24 MAX_UPLOAD_MB=25 MAX_CONCURRENT=2 \
    MPLCONFIGDIR=/tmp/mpl GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY requirements.txt .
# mediapipe pulls in the GUI build of OpenCV (needs libGL) and JAX (unused, ~350 MB):
# swap in headless OpenCV and drop JAX.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-contrib-python opencv-python jax jaxlib && \
    pip install --no-cache-dir --no-deps --force-reinstall opencv-contrib-python-headless==4.11.0.86 && \
    python -c "import cv2, mediapipe, skimage, shapely, trimesh, manifold3d; mediapipe.solutions.face_mesh"

COPY app/ /app/
RUN useradd -u 99 -g 100 -M -s /usr/sbin/nologin app && \
    mkdir -p /data && chown -R 99:100 /data

USER 99:100
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
