from sample_client_api.nvidia.client.nvidia_request import NvidiaRequest


class NvidiaImageGenerationClientException(Exception):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
        custom_msg: str = None,
    ):
        self.nvidia_request = nvidia_request
        self.task_id = task_id
        self.url = url
        message = f"Failed request: {task_id}: {nvidia_request} to {url} with response: {text}, status_code: {status}"
        if custom_msg is not None:
            message = f"{message} and {custom_msg}"
        super().__init__(message)


class NvidiaPollException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NSFWRejectionException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NSFWRejectionFaceswapException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NSFWRejectionSDXLException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NvidiaFunctionNotFoundException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NvidiaOOMException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
    ):
        super().__init__(nvidia_request, task_id, url, status, text)


class NvidiaPostClientException(NvidiaImageGenerationClientException):
    def __init__(
        self,
        nvidia_request: NvidiaRequest,
        task_id: str,
        url: str,
        status: int,
        text: str,
        payload: str,
    ):
        message = f" payload : {payload}"
        super().__init__(nvidia_request, task_id, url, status, text, message)
