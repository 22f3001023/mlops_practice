#!/bin/bash
set -e

echo "=== 1. Activating Environment and Setting GCP Credentials ==="
source venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/.secrets/gcp_credentials.json

# Define a version for this run (e.g., using a timestamp or a simple v1)
DATA_VERSION="v1.0"
RUN_TAG="run-${DATA_VERSION}"

echo "=== 2. Pulling Latest Data from DVC ==="
dvc pull -r gcs

echo "=== 3. Reproducing DVC Pipeline (Process, Apply Feast, Train) ==="
dvc repro

echo "=== 4. Materializing Features in Feast ==="
(cd feature_repo && python materialize.py)

echo "=== 5. Pushing Pipeline Outputs to DVC ==="
dvc push -r gcs

echo "=== 6. Committing and Pushing to GitHub ==="
git add .
git commit -m "Run pipeline and train model: ${RUN_TAG}"
echo "Commits pushed to GitHub."

echo "=== 7. Tagging Release and Pushing Tags ==="
git tag -a "${RUN_TAG}" -m "Completed pipeline run for ${RUN_TAG}"
git push origin main --tags
echo "Tags pushed to GitHub."

echo "=== Pipeline run ${RUN_TAG} complete. ==="
