#!/bin/bash
set -e

FUNCTION_NAME="luyin-daily-digest"
REGION="ap-southeast-1"
RULE_NAME="luyin-digest-daily"

echo "==> Getting Lambda ARN..."
LAMBDA_ARN=$(aws lambda get-function \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" \
    --query "Configuration.FunctionArn" \
    --output text)

echo "==> Creating EventBridge rule (cron: 01:00 UTC = 08:00 WIB)..."
aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "cron(0 1 * * ? *)" \
    --state ENABLED \
    --region "$REGION" \
    --output text

echo "==> Granting EventBridge permission to invoke Lambda..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Remove existing permission if it exists (re-runnable)
aws lambda remove-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id EventBridgeDailyTrigger \
    --region "$REGION" 2>/dev/null || true

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id EventBridgeDailyTrigger \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
    --region "$REGION" \
    --output text

echo "==> Attaching Lambda as EventBridge target..."
aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "Id=1,Arn=${LAMBDA_ARN}" \
    --region "$REGION"

echo ""
echo "==> EventBridge schedule active."
echo "==> Digest will post daily at 08:00 WIB (01:00 UTC)."
echo "==> Test with: aws lambda invoke --function-name $FUNCTION_NAME --region $REGION --payload '{}' /tmp/response.json && cat /tmp/response.json"
