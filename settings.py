from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
AI_IP = os.getenv("AI_IP")
PLATFORM_API_URL = os.getenv("PLATFORM_API_URL")
AI_SERVICE_SECRET = os.getenv("AI_SERVICE_SECRET_KEY")
AWS_DETECT_TEXT_ACCESS_KEY_ID = os.getenv("AWS_DETECT_TEXT_ACCESS_KEY_ID")
AWS_DETECT_TEXT_ACCESS_KEY = os.getenv("AWS_DETECT_TEXT_ACCESS_KEY")
AWS_DETECT_TEXT_REGION = os.getenv("AWS_DETECT_TEXT_REGION")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
## blob string
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_AI_CONTAINER_NAME = "aiapis"

# Azure GPT4 vision preview
AZURE_GPT_ENDPOINT = os.getenv("AZURE_GPT4_VISION_ENDPOINT")
AZURE_GPT_API_KEY = os.getenv("AZURE_GPT4_VISION_API_KEY")
AZURE_GPT_API_VERSION = os.getenv("AZURE_GPT4_VISION_API_VERSION")
AZURE_GPT_MODEL = os.getenv("AZURE_GPT4_VISION_MODEL")

# Platform Monitoring URLs + login

FINANCIAL_API_USERNAME = os.getenv("FINANCIAL_API_USERNAME")
FINANCIAL_API_PASSWORD = os.getenv("FINANCIAL_API_PASSWORD")
FINANCIAL_API_ORG = os.getenv("FINANCIAL_API_ORG")
FINANCIAL_API_URL_LOGIN = os.getenv("FINANCIAL_API_URL_LOGIN")

PUSH_API_URL = f"{PLATFORM_API_URL}/monitoring-v2/api/v1/document/postParsedData?"
FAIL_API_URL = f"{PLATFORM_API_URL}/monitoring-v2/api/v1/document/postParsingError?"
STATUS_API_URL = f"{PLATFORM_API_URL}/monitoring-v2/api/v1/document/setExtractionStarted?"

MAX_PLATFORM_UPLOAD_RETRIES = 5
RETRY_UPLOAD_WAIT_TIME = 2


BLOB_MODEL_DESTINATION = "model/best_downloaded_model.pt"
BLOB_NAME = "chart_parser_model"

BLOB_MODEL_DESTINATION_2 = "model/best_model_chart_detect.pt"
BLOB_NAME_2 = "chart_detect_model"

BLOB_MODEL_DESTINATION_3 = "model/best_model_classification.json"
BLOB_NAME_3 = "chart_classification"

BLOB_MODEL_DESTINATION_4 = "model/best_model_classification.h5"
BLOB_NAME_4 = "chart_classification_weight"

BLOB_MODEL_DESTINATION_5 = "model/bars_legends_best.pt"
BLOB_NAME_5 = "bars_legends_model_latest"

# Elastic search
ELASTIC_LOG_INDEX_NAME = "chart_parser_logs"
ELASTIC_DB_IP = os.getenv("ELASTIC_DB_IP")
ELASTIC_DB_USERNAME = os.getenv("ELASTIC_DB_USERNAME")
ELASTIC_DB_KEY = os.getenv("ELASTIC_DB_KEY")
ELASTIC_SCHEME = os.getenv("SCHEME")
ELASTIC_PORT = os.getenv("ELASTIC_PORT")

CHART_TYPES = {
    0: "AreaGraph",
    1: "BarGraph",
    2: "BoxPlot",
    3: "BubbleChart",
    4: "FlowChart",
    5: "LineGraph",
    6: "Map",
    7: "NetworkDiagram",
    8: "ParetoChart",
    9: "PieChart",
    10: "ScatterGraph",
    11: "TreeDiagram",
    12: "VennDiagram",
}


# RabbitMQ
RABBITMQ_IP = os.getenv("RABBITMQ_IP")
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

# MySQL
MYSQL_HOST_IP = os.getenv("MYSQL_HOST_IP")
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_SCHEMA = os.getenv("MYSQL_SCHEMA")

# main queues for UAT and production
CHART_PARSER_V2_QUEUE = "ai.chart_parser.queue"
HEADER_EXTRACTOR_QUEUE = "ai.header_extractor.queue"

# chart table
TABLE_NAME = "chart_jobs"


# SQL DB JOB STATUS VALUES
JOB_STATUS_STARTED = "STARTED"
JOB_STATUS_COMPLETED = "COMPLETED"

CHART_PARSER_V2_QUEUED = "QUEUED"
CHART_PARSER_V2_STARTED = "STARTED"
CHART_PARSER_V2_FINISHED = "FINISHED"

HEADER_EXTRACTION_DORMANT = "DORMANT"
HEADER_EXTRACTION_QUEUED = "QUEUED"
HEADER_EXTRACTION_STARTED = "STARTED"
HEADER_EXTRACTION_FINISHED = "FINISHED"

LOCAL = False  # CHANGE THIS TO TRUE WHILE TESTING ON LOCAL
if LOCAL:
    # queue for local testing
    CHART_PARSER_V2_QUEUE = "hello_chart_parser"
    HEADER_EXTRACTOR_QUEUE = "hello_header_extractor"
    TABLE_NAME = "chart_jobs"

# Azure openai key
AZURE_GPT_4O_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")
AZURE_GPT_4O_API_VERSION = os.getenv("AZURE_GPT4O_API_VERSION")
AZURE_GPT_4O_ENDPOINT = os.getenv("AZURE_GPT4O_ENDPOINT")
AZURE_GPT_4O_DEPLOYMENT = os.getenv("AZURE_GPT4O_MODEL")

# claude secret key
AWS_SECRET_KEY = os.getenv("AWS_BEDROCK_SECRET_KEY")
AWS_KEY = os.getenv("AWS_BEDROCK_ACCESS_KEY_ID")
AWS_REGION = os.getenv("AWS_BEDROCK_REGION")
