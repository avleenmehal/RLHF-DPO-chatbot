#!/bin/bash
# Usage: ./deploy.sh
# Builds on GCP (no local cross-compile) and redeploys Cloud Run.

set -e

PROJECT=medical-chat-prod
IMAGE=us-central1-docker.pkg.dev/$PROJECT/medchat-repo/medchat:latest
REGION=us-central1

echo "==> Building image on Cloud Build..."
gcloud builds submit --tag $IMAGE --project $PROJECT

echo "==> Deploying to Cloud Run..."
gcloud run deploy medchat \
  --image=$IMAGE \
  --platform=managed \
  --region=$REGION \
  --service-account=medchat-sa@$PROJECT.iam.gserviceaccount.com \
  --add-cloudsql-instances=$PROJECT:$REGION:medchat-db \
  --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest,JWT_SECRET=JWT_SECRET:latest,DB_PASSWORD=DB_PASSWORD:latest,NEO4J_PASSWORD=NEO4J_PASSWORD:latest,NEO4J_URI=NEO4J_URI:latest,NEO4J_USERNAME=NEO4J_USERNAME:latest,DB_ENCRYPTION_KEY=DB_ENCRYPTION_KEY:latest \
  --set-env-vars="VECTOR_STORE_PATH=gs://$PROJECT-assets/vector_store,GCP_PROJECT=$PROJECT,DATABASE_URL=postgresql+psycopg2://medchat-user:medchatpassword@/medchat?host=/cloudsql/$PROJECT:$REGION:medchat-db" \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=2 \
  --project=$PROJECT

echo "==> Done. Service URL:"
gcloud run services describe medchat --region=$REGION --project=$PROJECT --format="value(status.url)"
