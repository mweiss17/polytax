import os
from google.cloud import storage

def _upload_blob_gcs(local_path, gcs_path):
   """Uploads a file to GCS bucket"""
   client = storage.Client()
   blob = storage.blob.Blob.from_string(gcs_path)
   blob.bucket._client = client
   blob.upload_from_filename(local_path)

def _read_blob_gcs(bucket, checkpoint_file, destination):
   """Downloads a file from GCS to local directory"""
   client = storage.Client()
   bucket = client.get_bucket(bucket)
   blob = bucket.get_blob(checkpoint_file)
   blob.download_to_filename(destination)