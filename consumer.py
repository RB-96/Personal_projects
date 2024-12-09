from settings import (
    RABBITMQ_IP,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    CHART_PARSER_V2_STARTED,
    HEADER_EXTRACTOR_QUEUE,
    HEADER_EXTRACTION_QUEUED,
    CHART_PARSER_V2_QUEUE,
    CHART_PARSER_V2_FINISHED,
)

import base64
import pika
import json
import traceback
import threading
import time
from datetime import datetime
import os
import shutil
import pandas as pd


from elastic_logging import upload_to_elasticsearch, update_in_elasticsearch
from db_helper import fetch_job_from_job_id, update_job_from_job_id, insert_job_in_db
from blob_helper import fetch_blob_from_name, upload_blob_with_name
from server import check_secret_key, clear_data_folder, allowed_file
from bar_charts_handler.bar_legend_model.bar_legend_model_downloader import (
    model_downloader,
)
from chart_classification.classification_model_download import (
    classification_model_downloader,
)
from app.server.services.chart_extract_pipeline import ChartParsing
import chart_classification.classification_type as cls_type
from publisher import publish_to_queue
from platform_utils import update_extraction_status


def work_wrapper(body):
    """starts a chart parser job in the queue

    Args:
        body (_type_): message received from queue
    """
    queue_logging = []
    job_payload = json.loads(body.decode())
    liveness = job_payload.get("liveness")
    blob_name_file = job_payload.get("blob_name")
    job_id = ""

    try:
        job_id = job_payload.get("job_id")
    except Exception as e:
        queue_logging.append({"message": f"unable to parse queue message: {job_payload}"})

    if liveness:
        status_db, traceback_status_db = insert_job_in_db(
            job_id=job_id,
            model_used="LIVENESS",
            datetime_created=datetime.now(),
        )
        if not status_db:
            print("queue liveness. create row in db failed")
            print(f"traceback {traceback_status_db}")
        return

    if not job_id:
        message = "job_id not found"
        print(message)
        queue_logging.append(
            {
                "error": message,
            }
        )
    else:
        t_queue = time.time()
        (
            update_chart_parser_v2_status,
            traceback_chart_parser_v2_status,
        ) = update_job_from_job_id(job_id, "chart_parser_v2_status", CHART_PARSER_V2_STARTED)

        if not update_chart_parser_v2_status:
            queue_logging.append(
                {
                    "message": "update chart_parser_v2_status in db failed",
                    "traceback": traceback_chart_parser_v2_status,
                }
            )

        path_job_id = os.path.join(os.path.join(os.getcwd(), "pdfs_queue"), f"request_{job_id}")
        os.mkdir(path_job_id)

        blob_status, payload_serialized = fetch_blob_from_name(blob_name=blob_name_file)

        if blob_status:
            payload = json.loads(payload_serialized)

            file = payload["file"]
            file_name = payload["file_name"]
            request_id = payload["request_id"]
            secret_id = payload["secret_id"]
            company_name = payload["company_name"]
            fiscal_year = payload["fiscal_year"]
            documentId = payload["documentId"]
            orgId = payload["orgId"]
            unit = payload["unit"]
            currency = payload["currency"]
            human_language = payload["human_language"]

            log_data = {
                "documentId": documentId,
                "job_id": job_id,
                "details": [],
                "extraction_status": "NotStarted",
                "extraction_time": "NotChecked",
                "header_extraction_status": "NotStarted",
            }

            secret_id_status, secret_id_message = check_secret_key(secret_id)
            if not secret_id_status:
                print("Secret ID mismatch")
                return

            status_update_extraction = update_extraction_status(documentId)
            log_data["extraction_status"] = "Started"

            resp = upload_to_elasticsearch(log_data)
            try:
                es_id = resp["_id"]
            except:
                print("ELK STACK DOWN")
                es_id = "DummyEsId"

            file = base64.b64decode(file)
            model_used = None
            c_type = None
            title = ""
            chart_type = None

            if file and allowed_file(file_name):
                pdf_path = os.path.join(path_job_id, "image_folder")
                os.mkdir(pdf_path)

                file_path = os.path.join(pdf_path, file_name)

                with open(file_path, "wb+") as file_obj:
                    file_obj.write(file)

                log_data["extraction_status"] = "Started"
                update_in_elasticsearch(es_id, log_data)

                parser = ChartParsing(
                    file_name=file_name,
                    file_path=file_path,
                    path_job_id=path_job_id,
                    es_id=es_id,
                    log_data=log_data,
                )

                t_start = time.time()

                final_output = parser.parse_pdf()

                blob_extraction_payload = {"extraction": final_output}

                final_output_serialized = json.dumps(blob_extraction_payload)
                chart_parser_V2_blob_name = f"chart_parser_v2/{job_id}.json"
                blob_status = upload_blob_with_name(
                    blob_name=chart_parser_V2_blob_name,
                    serialized_data=final_output_serialized,
                )

                if blob_status:
                    (
                        update_db_blob_path_chart_parser_v2_status,
                        traceback_blob_path_chart_parser_v2_status,
                    ) = update_job_from_job_id(
                        job_id,
                        "blob_path_chart_parser_v2",
                        chart_parser_V2_blob_name,
                    )
                    if not update_db_blob_path_chart_parser_v2_status:
                        queue_logging.append(
                            {
                                "message": "update blob_path_chart_parser_v2 in db fail",
                                "traceback": traceback_blob_path_chart_parser_v2_status,
                            }
                        )

                    (
                        update_model_used_status,
                        traceback_model_used_status,
                    ) = update_job_from_job_id(
                        job_id,
                        "model_used",
                        model_used,
                    )
                    if not update_model_used_status:
                        queue_logging.append(
                            {
                                "message": "update model_used_status in db failed",
                                "traceback": traceback_model_used_status,
                            }
                        )

                    (
                        update_header_extraction_status,
                        traceback_header_extraction_status,
                    ) = update_job_from_job_id(
                        job_id,
                        "header_extraction_status",
                        HEADER_EXTRACTION_QUEUED,
                    )
                    if not update_header_extraction_status:
                        queue_logging.append(
                            {
                                "message": "update header_extraction_status in db failed",
                                "traceback": traceback_header_extraction_status,
                            }
                        )

                    message = {
                        "job_id": job_id,
                        "job_type": "chart",
                        "blob_name_file": blob_name_file,
                        "blob_name_extraction": chart_parser_V2_blob_name,
                    }
                    message_serialized = json.dumps(message)

                    status_publish = publish_to_queue(
                        queue=HEADER_EXTRACTOR_QUEUE,
                        message=message_serialized,
                    )

                    if not status_publish:
                        queue_logging.apppend({"message": "publish to header_extraction queue failed"})
                    log_data["blob_upload_status"] = f"Success {time.time() - t_start}"
                else:
                    queue_logging.append({"message": "pdf_extraction response blob upload failed"})

                log_data["queue_total_time_taken"] = f"{time.time() - t_queue}"
                if queue_logging:
                    log_data["queue_logging"] = queue_logging

                (
                    update_chart_parser_v2_status,
                    traceback_chart_parser_v2_status,
                ) = update_job_from_job_id(job_id, "chart_parser_v2_status", CHART_PARSER_V2_FINISHED)

                if not update_chart_parser_v2_status:
                    queue_logging.append(
                        {
                            "message": "update chart_parser_v2_status in db failed",
                            "traceback": traceback_chart_parser_v2_status,
                        }
                    )

                (
                    update_datetime_chart_finish,
                    traceback_datetime_chart_finish,
                ) = update_job_from_job_id(
                    job_id,
                    "datetime_chart_finish",
                    datetime.now(),
                )

                if not update_datetime_chart_finish:
                    log_data["chart_parser_update_finish_time_failure"] = (
                        f"update datetime_chart_finish in db failed. traceback: {traceback_datetime_chart_finish}"
                    )

                log_data['"header_extraction_status"'] = f"Pushed to queue"

                update_in_elasticsearch(es_id, log_data)
        else:
            log_data["blob_fetch_failure"] = f"error in fetching from blob. traceback: {payload_serialized}"

        shutil.rmtree(path_job_id)


def callback(ch, method, properties, body):
    """callback function triggered on reading message from queue

    Args:
        ch (_type_): _description_
        method (_type_): _description_
        properties (_type_): _description_
        body (_type_): message read from queue
    """
    print("Starting thread...")
    body_t = threading.Thread(target=work_wrapper, args=(body,))
    body_t.start()


def listen_to_rabbitmq(queue: str):
    """rabbitmq queue listener

    Args:
        queue (str): queue to listen to
    """
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_IP,
            port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials(username=RABBITMQ_USERNAME, password=RABBITMQ_PASSWORD),
        )
    )

    channel = connection.channel()
    channel.queue_declare(queue=queue)

    channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)

    print(f"consuming from queue: {queue}")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Consumption interrupted. Exiting...")
        connection.close()


if __name__ == "__main__":
    try:
        print("STARTING TO LISTEN TO QUEUE")
        listen_to_rabbitmq(CHART_PARSER_V2_QUEUE)
    except Exception as e:
        traceback_ex = traceback.format_exc()
        es_logging = {
            "CHART_PARSER_V2": {
                "message": "disconnected from rabbitmq",
                "traceback": traceback_ex,
            }
        }
        print(es_logging)
        upload_to_elasticsearch(es_logging)
