from fastapi import Header, HTTPException
from starlette import status
from wombo_aws_boto.secrets import get_environment_secret_value

SSR_TOKEN = get_environment_secret_value("ssrToken", default="ssr-credentials")


def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    # Split the header into 'bearer' and the token part
    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )

    if token != SSR_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid or expired token : {token}",
        )

    # If the token is valid, return True or some user identity if needed
    return True
