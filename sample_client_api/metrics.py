from opentelemetry import metrics
from sample_client_api.log_handling import get_logger_for_file

logger = get_logger_for_file(__name__)

meter = metrics.get_meter("nvidia_picasso")

successful_tasks_counter = meter.create_counter(
    name="successful_tasks", description="Successful tasks"
)

failed_tasks_counter = meter.create_counter(
    name="failed_tasks", description="Failed tasks"
)

task_processing_time = meter.create_histogram(
    name="processing_time", description="Time taken to process tasks", unit="s"
)
