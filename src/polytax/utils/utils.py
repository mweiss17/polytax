import os
from typing import Dict, Callable, Tuple
import torch
import torch.nn as nn
from google.cloud import storage
from dataclasses import dataclass


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

def reduce_gradients(xla_found, is_multi_host, optimizer):
   # AllReduce the model gradients so we can step the global gradient
   if not xla_found:
      return
   import torch_xla.core.xla_model as xm

   xm.reduce_gradients(optimizer)

   if not is_multi_host:
      return

   # if self.is_master_ordinal:
   #     dist.all_reduce(param.grad.cpu() / dist.get_world_size())
   #     xm.all_reduce(xm.REDUCE_SUM, param.grad.to(self.device))
   # else:
   #     zeros = torch.zeros_like(param.grad)
   #     xm.all_reduce(xm.REDUCE_SUM, zeros)
