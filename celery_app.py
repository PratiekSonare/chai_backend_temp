import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv('REDIS_URL')

celery_app = Celery(
    "history_kpi_cache",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    timezone="Asia/Kolkata",
    enable_utc=False,
)
