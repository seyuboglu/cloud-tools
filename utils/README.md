# Changing Cluster Configurations

## Hippo cluster
```bash
gcloud config set project hai-gcp-hippo
gcloud container clusters get-credentials platypus-1 --zone us-west1-a
```

## Fine-Grained cluster
```bash
gcloud config set project hai-gcp-fine-grained
gcloud container clusters get-credentials cluster-1 --zone us-west1-a
```
