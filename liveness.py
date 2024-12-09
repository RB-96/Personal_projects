import requests
import sys
import urllib3
import os
import logging
import settings

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