#!/bin/sh

gcloud functions deploy inference \
    --region=us-central1 \
    --runtime=python39 \
    --memory=512MB \
    --trigger-http
