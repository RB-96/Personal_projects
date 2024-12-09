import os
from dotenv import load_dotenv
from app.response_schemas import schema
import settings
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import json
import requests
import time
import uuid

from bar_charts_handler.bar_legend_model.bar_legend_model_downloader import (
    model_downloader,
)
from app.server.services.chart_extract_pipeline import ChartParsing
from chart_classification.classification_model_download import (
    classification_model_downloader,
)
import elastic_logging

model_downloader()
classification_model_downloader()

status = elastic_logging.create_elasticsearch_index_if_doesnt_exist()


def check_secret_key(secret_id):
    try:
        if str(secret_id).strip() == settings.AI_SERVICE_SECRET:
            return True, "Secret Key Matched"
        else:
            return False, "Secret Key Does Not Match. Incorrect Key..."

    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False, message


def get_chart_headers(file, orig_pdf_path, request_id, es_id, log_data):
    print("header extraction api called!")
    with open("new_file.json", "w") as f:
        json.dump(file, f)

    files = {
        "file": open("new_file.json"),
        "orig_pdf": open(orig_pdf_path, "rb").read(),
    }
    values = {
        "secret_id": settings.AI_SERVICE_SECRET,
        "request_id": request_id,
    }

    chart_preprocessing_url = os.path.join(settings.AI_IP, "chart_postprocessing")
    r = requests.post(chart_preprocessing_url, files=files, data=values)
    print(r.text)
    if r.status_code != 200:
        err = Exception(f"Header Ext. Failed: {r.status_code}, {r.reason}")
        log_data['"header_extraction_status"'] = err
        elastic_logging.update_in_elasticsearch(es_id, log_data)

        raise Exception(f"Header Ext. Failed: {r.status_code}, {r.reason}")

    json_data = json.loads(r.text)

    return json_data


ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = FastAPI()


@app.post("/chart_parser_v2", responses=schema.CHART_PARSER_RESPONSE)
async def chart_parser_v2(
    file: UploadFile, request_id: str = Form(...), secret_id: str = Form(...), documentId: str = Form(None)
):
    if not check_secret_key(secret_id):
        return Response(status_code=401, content="Incorrect Secret Key")

    log_data = {
        "documentId": documentId,
        "details": [],
        "extraction_status": "NotStarted",
        "extraction_time": "NotChecked",
        "header_extraction_status": "NotStarted",
    }

    resp = elastic_logging.upload_to_elasticsearch(log_data)

    try:
        es_id = resp["_id"]
    except:
        print("ELK STACK DOWN")
        es_id = "DummyEsId"

    logging_info = []

    job_id = uuid.uuid4()

    path_job_id = os.path.join(os.path.join(os.getcwd(), "pdfs_queue"), f"request_{job_id}")
    os.mkdir(path_job_id)

    if file and allowed_file(file.filename):
        try:
            pdf_path = os.path.join(path_job_id, "image_folder")
            os.mkdir(pdf_path)
        except:
            pass

        file_path = os.path.join(pdf_path, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log_data["extraction_status"] = "Started"
        elastic_logging.update_in_elasticsearch(es_id, log_data)
        parser = ChartParsing(
            file_name=file.filename, file_path=file_path, path_job_id=path_job_id, es_id=es_id, log_data=log_data
        )

        final_output = parser.parse_pdf()
        t_start = time.time()
        header_included = get_chart_headers(final_output, file_path, request_id, es_id, log_data)
        log_data['"header_extraction_status"'] = f"Success {time.time() - t_start}"
        elastic_logging.update_in_elasticsearch(es_id, log_data)

        clear_data_folder("image_folder")
        clear_data_folder("cropped_objects")
        clear_data_folder("pdf_images")

    return JSONResponse(status_code=200, content=header_included)
    # return header_included


@app.get("/chart_parser_v2/health_check", responses=schema.HEALTH_CHECK_RESPONSE)
async def health_check():
    return JSONResponse(status_code=200, content={"message": "Alive..."})


def clear_data_folder(directory):
    try:
        files = os.listdir(directory)
        print(files)
        for filename in files:
            if filename != "__init__.py":
                os.remove(os.path.join(directory, filename))
        return True
    except Exception as e:
        print(e)
        return False


def main():
    uvicorn.run("server:app", host="0.0.0.0", port=5900, reload=True)


if __name__ == "__main__":
    main()
