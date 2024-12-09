import pika
import json
import traceback
from datetime import datetime
from settings import RABBITMQ_IP, RABBITMQ_PASSWORD, RABBITMQ_PORT, RABBITMQ_USERNAME


def publish_to_queue(queue: str, message: str):
    """publishes a message to the specified queue

    Args:
        queue (str):queue name
        message (str): message to be published

    Returns:
        _type_: _description_
    """
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_IP,
                port=RABBITMQ_PORT,
                credentials=pika.PlainCredentials(
                    username=RABBITMQ_USERNAME, password=RABBITMQ_PASSWORD
                ),
            )
        )
        channel = connection.channel()
        channel.queue_declare(queue=queue)

        channel.basic_publish(exchange="", routing_key=queue, body=message)
        print("message published...")
        connection.close()
        return True, ""
    except Exception:
        print(f"Exception while publishing to queue {queue}")
        traceback_ex = traceback.format_exc()
        return False, traceback_ex
