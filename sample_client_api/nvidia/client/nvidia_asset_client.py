import asyncio
import json
from typing import Dict, Any, List

import aiohttp
from wombo_utilities import get_logger_for_file

from sample_client_api.nvidia.client.nvidia_request import NvidiaRequest, AssetLoader
from sample_client_api.nvidia.nvidia_token_manager import NvidiaAuthTokenManager

logger = get_logger_for_file(__name__)


class NvidiaAssetException(Exception):
    def __init__(self, message: str, status: int, text: str, url: str):
        final = message + f"url:{url}, status_code: {status}, response: {text}"
        super().__init__(final)


class NvidiaAssetCreationException(NvidiaAssetException):
    def __init__(self, field_name: str, status: int, text: str, url: str):
        super().__init__(f"Field: {field_name} ", status, text, url)


class NvidiaAssetUploadException(NvidiaAssetException):
    def __init__(self, asset_id: str, status: int, text: str, url: str):
        message = f"AssetID: {asset_id} "
        super().__init__(message, status, text, url)


class NvidiaAssetDeleteException(NvidiaAssetException):
    def __init__(self, asset_id: str, status: int, text: str, url: str):
        message = f"AssetID: {asset_id} "
        super().__init__(message, status, text, url)


def is_response_status_valid(response: aiohttp.ClientResponse) -> bool:
    return 200 <= response.status < 300


class NvidiaAssetClient:
    def __init__(self, token_manager: NvidiaAuthTokenManager, endpoint: str):
        logger.info("Initializing NvidiaAssetClient...")
        self.token_manager = token_manager
        self.endpoint = endpoint

    async def upload_asset(
        self,
        session: aiohttp.ClientSession,
        asset_loader: AssetLoader,
        token: str,
        field_name: str,
        data: Dict[str, Any],
    ) -> str:
        with await asset_loader() as asset:
            url = f"{self.endpoint}/assets"
            request = session.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                data=json.dumps(
                    {
                        "contentType": asset.content_type,
                        "description": field_name,
                    }
                ),
            )

            async with request as response:
                if not is_response_status_valid(response):
                    raise NvidiaAssetCreationException(
                        field_name, response.status, await response.text(), url
                    )

                res_json = await response.json()

            asset_id: str = res_json["assetId"]
            url = res_json["uploadUrl"]

            headers = {
                "Content-Type": res_json["contentType"],
                "x-amz-meta-nvcf-asset-description": res_json["description"],
                "Content-Length": str(asset.content_length),
            }

            async with session.put(
                url,
                headers=headers,
                data=asset.data,
            ) as response:
                if not is_response_status_valid(response):
                    raise NvidiaAssetUploadException(
                        asset_id, response.status, await response.text(), url
                    )

            data["requestBody"]["inputs"].append(
                {
                    "name": field_name,
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [asset_id],
                }
            )

            return asset_id

    async def delete_asset(
        self, session: aiohttp.ClientSession, asset_id: str, token: str
    ):
        logger.info(f"Deleting asset {asset_id}")
        url = f"{self.endpoint}/assets/{asset_id}"

        async with session.delete(
            url,
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            if not is_response_status_valid(response):
                raise NvidiaAssetDeleteException(
                    asset_id, response.status, await response.text(), url
                )

    async def cleanup_assets(
        self, session: aiohttp.ClientSession, assets: List[str], token: str
    ):
        if len(assets) <= 0:
            return

        await asyncio.gather(
            *[self.delete_asset(session, asset, token) for asset in assets]
        )

    async def handle_assets(
        self,
        session: aiohttp.ClientSession,
        nvidia_request: NvidiaRequest,
        token: str,
        data,
    ):
        tasks = [
            self.upload_asset(
                session,
                image,
                token,
                field,
                data,
            )
            for field, image in nvidia_request.assets.items()
            if image is not None
        ]

        assets: List[str] = [*await asyncio.gather(*tasks)]

        if len(assets) > 0:
            data["requestHeader"] = {"inputAssetReferences": assets}

        return assets, data
