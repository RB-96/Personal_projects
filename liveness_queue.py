import requests
import sys
import urllib3
import os
import logging
import settings
import uuid
import json
from publisher import publish_to_queue
import time
from db_helper import fetch_job_from_job_id, delete_job_from_job_id


logger = logging.getLogger()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

average_load_value = False
try:
    # fetching the Average load value usage in %
    average_load_value = os.getloadavg()[0] / os.cpu_count() * 100
    print(average_load_value)
    logger.info(f"fetched Average Load value: {average_load_value}")
except Exception as e:
    logger.error(f"Error fetching Average Load Usage value, {e}")

health_api_response = False
try:
    # sending /health request
    health_check_url = os.path.join(
        os.path.join(settings.AI_IP, "chart_parser_v2"), "health_check"
    )
    health_api_response = requests.get(health_check_url, verify=False)
    print(health_api_response)
    logger.info(f"got /health response: {health_api_response.status_code}")
except Exception as e:
    logger.error(f"could not get /health response, {e}")


queue_response = False
try:
    job_id = str(uuid.uuid4())
    message = {"job_id": job_id, "liveness": True}
    message_serialized = json.dumps(message)
    status_publish, traceback_queue_publish = publish_to_queue(
        queue=settings.CHART_PARSER_V2_QUEUE, message=message_serialized
    )

    if status_publish:
        print(f"Published")
        n_queue_liveness = 0
        while n_queue_liveness < 3:
            time.sleep(3 * (n_queue_liveness + 1))
            status_fetch, row = fetch_job_from_job_id(job_id=job_id)

            if status_fetch and row:
                print("Fetched liveness job_id from db")
                status_delete, traceback_delete = delete_job_from_job_id(job_id=job_id)
                if not status_delete:
                    print(f"ERROR deleting row with job_id {job_id} from db")
                    print(f"traceback: \n{traceback_delete}")
                else:
                    queue_response = True
                    print("Deleted liveness job_id from db")
                break
            else:
                print(f"ERROR fetching liveness row with job_id {job_id} from db")
                print(f"traceback: \n{row}")
            n_queue_liveness += 1
    else:
        print(
            f"error publishing liveness message with job_id {job_id} to email_attachment_queue {settings.CHART_PARSER_V2_QUEUE}"
        )
        print(f"traceback: \n {traceback_queue_publish}")

except Exception as e:
    logger.error(f"could not get queue liveness response, {e}")

# init tests state
average_load_fail = False
api_test_fail = False

# CPU average load validation
if not average_load_value:
    average_load_fail = True
    logger.error(f"Failed to fetch Average Load value, the test has been failed!")

elif average_load_value >= 90.0:
    average_load_fail = True
    logger.error(f"Average Load value is above threshold, the test has been failed!")
else:
    logger.debug(f"Success Average Load test")

# /health validation
if not health_api_response:
    api_test_fail = True
    logger.error(f"Could not get /health response, the test has been failed!")
elif health_api_response.status_code != 200:
    api_test_fail = True
    logger.error(f"Bad /health response, the test has been failed!")
elif health_api_response.status_code == 200:
    logger.debug(f"Success /health API test,")

# final validation
if api_test_fail:
    logger.error(f"Fail /health API test, Liveness probe has failed!, exiting")
    sys.exit(1)
# if average_load_fail:
#     logger.error(f"Fail Average load test, Liveness probe has failed!, exiting")
#     sys.exit(1)

logger.info(f"all tests passed, Liveness probe has succeed!, exiting")
sys.exit(0)
