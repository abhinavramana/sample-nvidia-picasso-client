from wombo_utilities import get_logger_for_file

from wombo import config
from wombo.nvidia.nvidia_task_handler import NvidiaImageGenerationTaskHandler
from wombo.nvidia.nvidia_token_manager import NvidiaAuthConfig

logger = get_logger_for_file(__name__)


def initialize_nvidia_service() -> NvidiaImageGenerationTaskHandler:
    logger.info("Initializing NVIDIA service...")

    auth_config = NvidiaAuthConfig(
        auth_url=config.NVCF_AUTH_URL,
        nvidia_username=config.NVIDIA_USERNAME,
        # We earlier had same secret for all environments, but now we have different secrets for different environments
        # nvidia_client_secret=get_secret_value(config.NVIDIA_CLIENT_SECRET, forced=True),
        nvidia_client_secret=config.NVIDIA_PASSWORD_TO_RENEW_90_DAYS,
        token_refresh_buffer_in_seconds=config.NVCF_TOKEN_REFRESH_BUFFER_IN_SECONDS,
    )

    nvidia_task_handler = NvidiaImageGenerationTaskHandler(
        nvcf_url=config.NVCF_URL, auth_config=auth_config
    )

    logger.info("Initialized NVIDIA service")
    return nvidia_task_handler


class BootupManager:
    def __init__(self):
        self.nvidia_task_handler: NvidiaImageGenerationTaskHandler = None
        self.perform_bootup()

    def perform_bootup(self):
        self.nvidia_task_handler = initialize_nvidia_service()


IMMUTABLE_BOOTUP_MANAGER = BootupManager()
