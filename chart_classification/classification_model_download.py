from azure.storage.blob import BlobServiceClient, BlobClient
from pathlib import Path
import os
import settings


def classification_model_downloader():
    if not os.path.exists("model"):
        os.makedirs("model", exist_ok=True)
    destination = settings.BLOB_MODEL_DESTINATION_3

    destination2 = settings.BLOB_MODEL_DESTINATION_4

    new_blob_name = settings.BLOB_NAME_3
    new_blob_name2 = settings.BLOB_NAME_4

    AZURE_CONNECTION_STRING = settings.AZURE_STORAGE_CONNECTION_STRING
    container_name = "aiapis"

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    if not Path(destination).exists():
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=new_blob_name)

        blob_data = blob_client.download_blob().readall()
        # Save the model data to a local file
        with open(destination, "wb") as file:
            file.write(blob_data)

        print("JSON downloaded successfully")

    if not Path(destination2).exists():
        print("Model is being downloaded...")
        blob_client2 = blob_service_client.get_blob_client(container=container_name, blob=new_blob_name2)

        blob_data2 = blob_client2.download_blob().readall()
        # Save the model data to a local file
        with open(destination2, "wb") as file:
            file.write(blob_data2)

        print("Weights (HDF5) downloaded successfully")


# classification_model_downloader()
