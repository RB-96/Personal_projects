import requests
from traceback import format_exc
from settings import (
    FINANCIAL_API_ORG,
    FINANCIAL_API_PASSWORD,
    FINANCIAL_API_URL_LOGIN,
    FINANCIAL_API_USERNAME,
    STATUS_API_URL,
    PUSH_API_URL,
    FAIL_API_URL,
    MAX_PLATFORM_UPLOAD_RETRIES,
    RETRY_UPLOAD_WAIT_TIME,
)
from time import sleep


def make_login_request():
    body = {
        "username": FINANCIAL_API_USERNAME,
        "password": FINANCIAL_API_PASSWORD,
        "orgType": FINANCIAL_API_ORG,
    }
    r = requests.post(
        url=FINANCIAL_API_URL_LOGIN,
        json=body,
    )
    response = r.headers
    X_AUTH = response["X-AUTH-TOKEN"]
    return X_AUTH


def update_extraction_status(docId):
    if docId:
        url = f"{STATUS_API_URL}documentId={docId}"
        print(url)
        headers = {"X-AUTH-TOKEN": make_login_request()}

        response = requests.get(
            url,
            headers=headers,
        )
        print(response.json())
        return True
    else:
        return False


def push_data_success(final_response, docId, orgId):

    retries = 0
    upload_status = False
    retries_resp_code = []

    url = f"{PUSH_API_URL}documentId={docId}&orgId={orgId}"
    headers = {"X-AUTH-TOKEN": make_login_request()}

    while not upload_status and retries < MAX_PLATFORM_UPLOAD_RETRIES:

        response = requests.post(
            url,
            headers=headers,
            json=final_response,
        )
        status_code = response.status_code

        if status_code == 200:
            upload_status = True
            return status_code, retries_resp_code

        else:
            retries_resp_code.append(
                {"response_code": status_code, "response": response.text}
            )
            retries += 1
            sleep(RETRY_UPLOAD_WAIT_TIME)

    return status_code, retries_resp_code


def push_data_fail(final_response, docId, orgId):
    url = f"{FAIL_API_URL}documentId={docId}&orgId={orgId}"

    headers = {"X-AUTH-TOKEN": make_login_request()}

    response = requests.post(url, headers=headers, json=final_response)

    print(response.json())

    return response.status_code
