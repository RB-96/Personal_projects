from azure.storage.blob import BlobServiceClient, BlobClient
import json
import traceback
from settings import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_AI_CONTAINER_NAME


def upload_blob_with_name(blob_name, serialized_data):
    """_summary_

    Args:
        blob_name (str):  name of the file to save under json suffix
        serialized_data (data serialized in bytes): the data to save in bytes
    Returns:
        bool:True or False depending on success state
    """
    # serialized_data = json.dumps(data)
    ### no need for extension to save bytes but useful to label it as a json request
    try:
        blob_client = BlobClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            container_name=AZURE_STORAGE_AI_CONTAINER_NAME,
            blob_name=blob_name,
        )
        blob_client.upload_blob(serialized_data)
        return True
    except Exception:
        print(f"Exception while uploading job {blob_name} and serialized data to blob")
        traceback.print_exc()
        return False


def fetch_blob_from_name(blob_name):
    """_summary_

    Args:
        blob_path ( str): str referring to the name of the blob
    Returns:
        json: json data deserilaized
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_AI_CONTAINER_NAME, blob=blob_name
        )
        data_downloaded_serialized = blob_client.download_blob().readall()
        return True, data_downloaded_serialized
    except Exception:
        print(f"Exception while fetching blob job {blob_name}")
        traceback_ex = traceback.format_exc()
        return False, traceback_ex
