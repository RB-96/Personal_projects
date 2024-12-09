from datetime import datetime

from elasticsearch import Elasticsearch

import settings

print([settings.ELASTIC_DB_IP])
elk_search = Elasticsearch(
    [settings.ELASTIC_DB_IP],
    http_auth=(settings.ELASTIC_DB_USERNAME, settings.ELASTIC_DB_KEY),
    scheme=settings.ELASTIC_SCHEME,
    port=settings.ELASTIC_PORT,
    retry_on_timeout=True,
    timeout=3,
)


def create_elasticsearch_index_if_doesnt_exist():
    # try:
    #     elk_search.indices.create(index=settings.ELASTIC_LOG_INDEX_NAME)
    #     print("Logging index created successfully")
    # except:
    #     print("Logging index already exists")
    #     pass
    # return None
    """_summary_
        creates an index on the Elastic Search Database of 73Strings
    """
    try:
        mapping = {
            "mappings": {
                "properties": {
                    "body_logs": {"type": "flattened"},  
                    "date": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis",
                    },
                }
            }
        }

        elk_search.indices.create(index=settings.ELASTIC_LOG_INDEX_NAME, body=mapping)
        print(f"{settings.ELASTIC_LOG_INDEX_NAME} logging index created successfully")
    except Exception as e:
        print(e)
        print(f"{settings.ELASTIC_LOG_INDEX_NAME} logging index already exists")
        pass


def upload_to_elasticsearch(log_data):
    try:
        # if log_data["documentId"]:
        log_data["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = elk_search.index(
            index=settings.ELASTIC_LOG_INDEX_NAME,
            body=log_data,
        )
        print(response)
        return response
    except Exception as e:
        print(e)
    return True

def update_in_elasticsearch(_id, log_data):

    if _id!="DummyEsId":
        try:
            print("settings.ELASTIC_LOG_INDEX_NAME", settings.ELASTIC_LOG_INDEX_NAME)
            resp = elk_search.update(
                index=settings.ELASTIC_LOG_INDEX_NAME, 
                id=_id, 
                body={
                    "doc": log_data
                }
            )
            print("log_data updated for: ", _id)
            print(resp)
            return True
        except Exception as e:
            print("UPDATE TO ELASTIC SEARCH FAILED")
            print(e)
    return False