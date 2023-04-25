#!bin/bash

gsutil cp -r $MODEL_ARTIFACT_URI ./pipeline/inference/model

RUN torch-model-archiver -f \
  --model-name=$MODEL_DISPLAY_NAME \
  --version=1.0 \
  --serialized-file=/pipeline/inference/pytorch_model.bin \
  --handler=/pipeline/inference/predictions/handler.py \
  --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json" \
  --export-path=/pipeline/inference/model

echo "Building serving image..."
docker build -t $DEPLOY_IMAGE_URI -f

echo "Pushing serving image to Artifact registry..."
docker push $DEPLOY_IMAGE_URI

echo "Executing Vertex AI Model Training Pipelines..."
python ./pipeline/pipeline_train.py train-model \
    --project ${GCP_PROJECT_ID} \
    --region ${GCP_REGION} \
    --storage_bucket ${GCP_STORAGE_BUCKET} \
    --service_account ${GCP_SERVICE_ACCOUNT} \
    --pipeline_root ${DEPLOY_PIPELINE_ROOT} \
    --serving_image_uri ${DEPLOY_IMAGE_URI} \
    --model_artifact_uri ${MODEL_ARTIFACT_URI} \
    --endpoint_artifact_uri ${ENDPOINT_ARTIFACT_URI} \
    --machine_type ${TRAIN_MACHINE_TYPE} \
    --accelerator_type ${TRAIN_ACCELERATOR_TYPE} \
    --accelerator_count ${TRAIN_ACCELERATOR_COUNT} \
    --min_replica_count ${AUTOSCALING_NODES_MIN} \
    --max_replica_count ${AUTOSCALING_NODES_MAX} \
    --cpu_target_utilization ${AUTSCALING_TARGET_CPU} \
    --traffic_percentage ${TRAFFIC_PERCENTAGE}
