#!/bin/bash

# MaintIE Azure ML Training Deployment Script
# This script submits training jobs to Azure ML workspace

echo "🚀 Starting MaintIE Azure ML Training Deployment..."

# Check Azure CLI login
if ! az account show &> /dev/null; then
    echo "❌ Not logged into Azure CLI. Please run: az login"
    exit 1
fi

# Set Azure ML workspace context
RESOURCE_GROUP="azure-ml-uwa"
WORKSPACE_NAME="azure-ml-uwa-workspace"

echo "📋 Using Azure ML Workspace: $WORKSPACE_NAME in $RESOURCE_GROUP"

# Submit SpERT training job
echo "🎯 Submitting SpERT training job..."
SPERT_JOB_ID=$(az ml job create \
    --file maintie-spert-job.yml \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME \
    --query name -o tsv)

if [ $? -eq 0 ]; then
    echo "✅ SpERT training job submitted: $SPERT_JOB_ID"
    echo "🔗 Monitor at: https://ml.azure.com/experiments/maintie-spert-training"
else
    echo "❌ Failed to submit SpERT training job"
    exit 1
fi

# Submit REBEL training job
echo "🎯 Submitting REBEL training job..."
REBEL_JOB_ID=$(az ml job create \
    --file maintie-rebel-job.yml \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME \
    --query name -o tsv)

if [ $? -eq 0 ]; then
    echo "✅ REBEL training job submitted: $REBEL_JOB_ID"
    echo "🔗 Monitor at: https://ml.azure.com/experiments/maintie-rebel-training"
else
    echo "❌ Failed to submit REBEL training job"
    exit 1
fi

echo ""
echo "🎉 Training deployment complete!"
echo "📊 Training Timeline:"
echo "   • Setup: 30 minutes"
echo "   • SpERT Training: 18-24 hours"
echo "   • REBEL Training: 18-24 hours"
echo ""
echo "📈 Monitor progress:"
echo "   • Azure ML Studio: https://ml.azure.com"
echo "   • SpERT Job: $SPERT_JOB_ID"
echo "   • REBEL Job: $REBEL_JOB_ID"
echo ""
echo "📋 Next Steps:"
echo "   1. Monitor training progress in Azure ML Studio"
echo "   2. Validate model performance (target F1 ≥87%)"
echo "   3. Proceed to Step 4: Model Evaluation"
