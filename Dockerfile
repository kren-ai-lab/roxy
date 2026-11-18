# Runtime image of a released Roxy version, installed from PyPI (no source copied), so any past
# release can be rebuilt. Built and pushed by .github/workflows/publish-container.yml.
#   docker build --build-arg ROXY_VERSION=0.3.0 -t roxy:0.3.0 .
FROM python:3.14-slim

ARG ROXY_VERSION
ARG EXTRAS=""
LABEL org.opencontainers.image.source="https://github.com/kren-ai-lab/roxy" \
      org.opencontainers.image.description="Roxy: classical protein sequence descriptors for machine learning" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${ROXY_VERSION}"

# procps: Nextflow reads task metrics with `ps` inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${ROXY_VERSION}" \
    && pip install --no-cache-dir "roxylib${EXTRAS}==${ROXY_VERSION}"
