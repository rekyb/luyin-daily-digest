#!/bin/bash
set -e

FUNCTION_NAME="luyin-daily-digest"
REGION="ap-southeast-1"
RUNTIME="python3.12"
HANDLER="main.handler"
# Replace YOUR_ACCOUNT_ID with your actual AWS account ID after running:
# aws sts get-caller-identity --query Account --output text
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-basic-execution"

echo "==> Packaging Lambda function..."
rm -rf /tmp/lambda-package
mkdir -p /tmp/lambda-package

# Install dependencies into package dir (excludes dev dependencies)
pip install -r requirements.txt -t /tmp/lambda-package --quiet

# Copy all source files
cp main.py fetcher.py summarizer.py formatter.py publisher.py config.py sources.yaml /tmp/lambda-package/

# Zip the package
cd /tmp/lambda-package
zip -r /tmp/lambda-package.zip . -q
cd -

echo "==> Deploying to AWS Lambda..."
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file fileb:///tmp/lambda-package.zip \
        --region "$REGION" \
        --output text
    echo "==> Function code updated."
else
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --role "$ROLE_ARN" \
        --handler "$HANDLER" \
        --zip-file fileb:///tmp/lambda-package.zip \
        --timeout 120 \
        --memory-size 256 \
        --region "$REGION" \
        --output text
    echo "==> Function created."
fi

echo "==> Setting environment variables..."
aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --environment "Variables={GEMINI_API_KEY=${GEMINI_API_KEY},SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}}" \
    --region "$REGION" \
    --output text

echo ""
echo "==> Done. Lambda deployed: $FUNCTION_NAME in $REGION"
echo "==> Next: set up the EventBridge schedule by running deploy/setup_eventbridge.sh"
