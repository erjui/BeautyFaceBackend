#!/bin/sh

gcloud functions deploy inference \
    --region=asia-northeast1 \
    --runtime=python39 \
    --trigger-topic=every-1h \
    --memory=512MB \
