import requests

TOPIC = "qnlp_trainer"


def send_notification(message):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=message.encode(encoding="utf-8"), timeout=5)
    except Exception as e:
        import logging

        logging.getLogger("training_notifications").warning(f"Failed to send notification: {e}")


def send_training_finished_notification(training_metrics: dict):
    send_notification(f"Training completed! Results on test set: {training_metrics}")
