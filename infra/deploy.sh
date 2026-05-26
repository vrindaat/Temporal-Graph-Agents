#!/bin/bash
set -e

BUCKET=${1:?"Usage: ./deploy.sh <s3-bucket-name>"}

echo "Uploading graph to S3..."
aws s3 cp ../thesis_graph.pkl s3://$BUCKET/thesis_graph.pkl

echo "Building SAM application..."
sam build

echo "Deploying..."
sam deploy \
  --stack-name tga-api \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides GraphS3Bucket=$BUCKET \
  --resolve-s3

echo "Done! API URL:"
aws cloudformation describe-stacks \
  --stack-name tga-api \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text
