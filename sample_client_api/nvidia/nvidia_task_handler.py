from time import perf_counter
from typing import Optional, Dict, Tuple, List, Any

from fastapi import HTTPException
from starlette import status
from wombo_utilities import get_logger_for_file

from wombo.metrics import (
    failed_tasks_counter,
    task_processing_time,
    successful_tasks_counter,
)
from wombo.nvidia.client.nvidia_image_generation_client import (
    NvidiaImageGenerationClient,
)
from wombo.nvidia.client.nvidia_request import NvidiaRequest
from wombo.nvidia.nvidia_token_manager import NvidiaAuthConfig

logger = get_logger_for_file(__name__)


class NvidiaImageGenerationTaskHandler:
    def __init__(self, nvcf_url: str, auth_config: NvidiaAuthConfig):
        self.nvidia_client = NvidiaImageGenerationClient(nvcf_url, auth_config)

    async def handle_nvidia_task(
        self,
        nvidia_client_request: NvidiaRequest,
        task_id: str,
        metrics_attributes: Optional[Dict[str, str]] = None,
    ) -> Optional[Tuple[bytes, List[Any]]]:
        timer = perf_counter()
        results, reason_for_failure = await self.nvidia_client.generate_image(
            nvidia_client_request, task_id
        )
        time_taken = perf_counter() - timer

        if metrics_attributes:
            task_processing_time.record(time_taken, metrics_attributes)

        logger.info(f"Task {task_id} took ${time_taken:.2f}s")

        if results:
            if metrics_attributes:
                successful_tasks_counter.add(1, metrics_attributes)

            return results
        else:
            if metrics_attributes:
                failed_tasks_counter.add(1, metrics_attributes)

            raise HTTPException(
                detail=f"Task {task_id} failed due to {reason_for_failure} with request: {nvidia_client_request}",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
