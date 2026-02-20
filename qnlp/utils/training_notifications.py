import requests

TOPIC = "qnlp_trainer"

def send_notification(message):
    requests.post(
        f"https://ntfy.sh/{TOPIC}",
        data=message.encode(encoding='utf-8')
    )

def send_training_finished_notification(training_metrics: dict):
    send_notification(f"Training completed! Results on test set: {training_metrics}")