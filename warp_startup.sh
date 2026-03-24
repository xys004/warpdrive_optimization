#!/bin/bash
set -e
cd /tmp

# Debian 12 requires venv for pip installs (PEP 668)
apt-get update -q && apt-get install -y -q python3-venv python3-dev
python3 -m venv /tmp/warpenv
source /tmp/warpenv/bin/activate

pip install -q tensorflow==2.12.0 numpy

gsutil -m cp "gs://warpopt-data/code/*.py" .
mkdir -p input
gsutil -m cp "gs://warpopt-data/golden_dataset/*" input/

python vertex_compute_maps.py \
    --domain 2 --plane-n 2000 \
    --input-dir ./input \
    --output-dir ./output_d2

python vertex_compute_maps.py \
    --domain 1 --plane-n 2000 \
    --input-dir ./input \
    --output-dir ./output_d1

gsutil -m cp -r ./output_d2/* gs://warpopt-data/hires_maps/domain2/
gsutil -m cp -r ./output_d1/* gs://warpopt-data/hires_maps/domain1/

echo "ALL DONE" | gsutil cp - gs://warpopt-data/hires_maps/DONE.txt

ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
