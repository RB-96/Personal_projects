from azure.storage.blob import BlobServiceClient, BlobClient
from pathlib import Path

import os
import settings


def model_downloader():
    if not os.path.exists("model"):
        os.makedirs("model", exist_ok=True)

    destination = settings.BLOB_MODEL_DESTINATION_5
    new_blob_name = settings.BLOB_NAME_5

    AZURE_CONNECTION_STRING = settings.AZURE_STORAGE_CONNECTION_STRING
    container_name = "aiapis"

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    if not Path(destination).exists():
        print("Model is being downloaded...")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=new_blob_name)

        blob_data = blob_client.download_blob().readall()
        # Save the model data to a local file
        with open(destination, "wb") as file:
            file.write(blob_data)

        print("model downloaded successful")
