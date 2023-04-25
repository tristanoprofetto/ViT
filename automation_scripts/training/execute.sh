#!bin/bash

echo "Building training image..."
docker build -t $TRAIN_IMAGE_URI -f

echo "Pushing training image to Artifact registry..."
docker push $TRAIN_IMAGE_URI

echo "Executing Vertex AI Model Training Pipelines..."
python ./pipeline/pipeline_train.py train-model \
--project ${GCP_PROJECT_ID} \
--region ${GCP_REGION} \
--bucket ${GCP_STORAGE_BUCKET} \
--service_account ${GCP_SERVICE_ACCOUNT} \
--pipeline_root ${TRAIN_PIPELINE_ROOT} \
--train_image_uri ${TRAIN_IMAGE_URI} \
--machine_type ${TRAIN_MACHINE_TYPE} \
--accelerator_type ${TRAIN_ACCELERATOR_TYPE} \
--accelerator_count ${TRAIN_ACCELERATOR_COUNT} \
