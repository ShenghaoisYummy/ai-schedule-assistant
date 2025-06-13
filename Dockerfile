###############################################################################
# 1 ── Build the Flutter web bundle (unchanged, stays in its own stage)      #
###############################################################################
FROM ghcr.io/cirruslabs/flutter:stable AS flutter-build
WORKDIR /app/frontend
COPY frontend/pubspec.* ./
RUN flutter pub get
COPY frontend/ .
RUN flutter build web --release


###############################################################################
# 2 ── Build Python dependencies in a *throw-away* layer                     #
###############################################################################
FROM python:3.10-alpine AS python-build
WORKDIR /install
# Required system libs only – no compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
        libblas3 liblapack3 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
# '--prefix=/install' installs wheels outside site-packages
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip uninstall -y torch \
    && pip install --no-cache-dir --prefix=/install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir --prefix=/install gunicorn \
    && python -m pip cache purge


###############################################################################
# 3 ── Final runtime image (tiny)                                             #
###############################################################################
FROM python:3.10-alpine

# Copy just the built wheels (no pip, no cache, no headers, no gcc)
COPY --from=python-build /install /usr/local
# Copy your backend source
WORKDIR /app
COPY backend/ .

# Copy the already-built Flutter web bundle
COPY --from=flutter-build /app/frontend/build/web ./static

# Tiny health + API server (already patched)
COPY combined_app.py .

# Strip debug symbols from .so files (saves ~300 MB)
RUN find /usr/local -name '*.so' -exec strip --strip-unneeded {} + || true

# Clean up any __pycache__ / tests / dist-info RECORD files
RUN find /usr/local -name '__pycache__' -prune -exec rm -rf {} + \
 && find /usr/local -name 'tests'       -prune -exec rm -rf {} + \
 && find /usr/local -name '*.pyc' -delete

# Make sure gunicorn is available in the PATH
RUN pip install --no-cache-dir gunicorn

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "combined_app:app"]