#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS=/Users/rjtang/_hack/hear_it/speech_api/hear_it_33991c4599e2.json
gcloud auth activate-service-account --key-file=hear_it_33991c4599e2.json
access_token=$(gcloud auth application-default print-access-token)
echo "$access_token"

set -x
curl -s -H "Content-Type: application/json" \
    -H "Authorization: Bearer $access_token" \
    https://speech.googleapis.com/v1/speech:recognize \
    -d @req.json
