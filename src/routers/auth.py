from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.utils.logger import get_logger, log_execution_time_async
from src.utils.supabase_client import supabase

logger = get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


class SignUpRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str


class SignInRequest(BaseModel):
    email: str
    password: str


@log_execution_time_async
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Validate JWT token from Authorization header and return user
    The token should be the session token from Supabase client's auth.getSession()
    """
    try:
        # Get user from JWT token
        user = supabase.get_user_by_jwt(credentials.credentials)
        if not user:
            logger.warning("Invalid or expired token")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user's profile
        profile = supabase.get_profile(user.user.id)
        if not profile:
            logger.warning(f"Profile not found for user {user.user.id}")
            raise HTTPException(
                status_code=404,
                detail="User profile not found",
            )

        return {
            "id": profile["id"],
            "email": user.user.email,
            "first_name": profile["first_name"],
            "last_name": profile["last_name"],
        }
    except Exception as e:
        logger.error("Authentication error", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
