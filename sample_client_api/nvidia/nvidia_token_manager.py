from datetime import datetime, timezone
from typing import Optional

import aiohttp
import jwt
from aiohttp import BasicAuth
from pydantic import BaseModel
from wombo_utilities import get_logger_for_file
from wombo_utilities.api_utilities.ip_address_retrieval import (
    get_global_ip,
    get_local_ip,
)

logger = get_logger_for_file(__name__)

NVIDIA_AUTH_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}
NVIDIA_AUTH_PAYLOAD = {
    "grant_type": "client_credentials",
    "scope": "invoke_function list_functions queue_details",
}


class NvidiaTokenException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NvidiaAuthConfig(BaseModel):
    auth_url: str
    nvidia_client_secret: str
    nvidia_username: str
    token_refresh_buffer_in_seconds: int


class NvidiaAuthTokenManager:

    def __init__(self, nvidia_auth_config: NvidiaAuthConfig):
        self.global_ip = (
            get_global_ip()
        )  # This can change if the server is behind a load balancer
        self.local_ip = get_local_ip()  # Doesn't change till bootup
        logger.info(
            f"Initializing NvidiaTokenManager with GlobalIP: {self.global_ip}, LocalIP: {self.local_ip}"
        )
        self.nvidia_auth_config = nvidia_auth_config
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        self.token: Optional[str] = None

    def validate_token(self) -> bool:
        # ensure the token is not expired
        try:
            if self.token is None or self.token == "":
                return False
            # Decode the JWT
            decoded_token = jwt.decode(self.token, options={"verify_signature": False})
            # Check if the token has expired
            token_expiry_time = decoded_token["exp"]
            curr_time = datetime.now(timezone.utc).timestamp()
            if (
                token_expiry_time
                > curr_time + self.nvidia_auth_config.token_refresh_buffer_in_seconds
            ):
                return True
            return False
        except Exception as e:
            logger.error(
                f"Error decoding token: {self.token} due to {e}", exc_info=True
            )
            return False

    async def fetch_token_if_required(
        self, session: aiohttp.ClientSession, token: Optional[str] = None
    ) -> str:
        if token is not None:
            self.token = token
        is_valid_token = self.validate_token()
        if not is_valid_token:
            self.token = await self.get_auth_token(session)
        return self.token

    async def get_auth_token(self, session: aiohttp.ClientSession) -> str:
        self.global_ip = get_global_ip()
        logger.info(
            f"Getting new auth token for GlobalIP: {self.global_ip}, LocalIP: {self.local_ip}"
        )
        request = session.request(
            "POST",
            # TODO: nvidia_client_secret should also be retrieved from secrets manager dynamically so that we dont have to redeploy the service
            # https://linear.app/wombo/issue/BAK-221
            auth=BasicAuth(
                self.nvidia_auth_config.nvidia_username,
                self.nvidia_auth_config.nvidia_client_secret,
            ),
            url=self.nvidia_auth_config.auth_url,
            headers=NVIDIA_AUTH_HEADERS,
            data=NVIDIA_AUTH_PAYLOAD,
        )

        async with request as auth_response:
            try:
                token = (await auth_response.json())["access_token"]
            except Exception as e:
                raise NvidiaTokenException(
                    f"Error getting auth token: {auth_response.text} due to {e}"
                )
            return token
