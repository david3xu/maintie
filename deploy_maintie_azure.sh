#!/bin/bash
# deploy_maintie_azure.sh

echo "Deploying MaintIE to Azure ML..."

# Create Azure ML compute cluster
az ml compute create --name maintie-cpu-cluster \
  --type amlcompute \
  --size Standard_E16s_v3 \
  --max-instances 2 \
  --min-instances 0 \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Upload code to Azure ML
az ml data create --name maintie-code \
  --type uri_folder \
  --path ./ \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Submit SpERT training job
az ml job create --file azure-ml-spert-job.yml \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

echo "MaintIE deployment initiated. Monitor progress in Azure ML Studio."
