#!/bin/sh

gcloud functions deploy inference \
    --region=us-central1 \
    --runtime=python39 \
    --memory=2048MB \
    --trigger-http \
	--allow-unauthenticated
