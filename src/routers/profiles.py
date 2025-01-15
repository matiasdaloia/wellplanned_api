from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.routers.auth import get_current_user
from src.utils.supabase_client import supabase

router = APIRouter(prefix="/profile", tags=["profiles"])


@router.get("")
async def get_profile(user=Depends(get_current_user)):
    """Get the current user's profile"""
    profile = supabase.get_profile(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/image")
async def upload_profile_image(
    file: UploadFile = File(...), user=Depends(get_current_user)
):
    """Upload a profile image and update the user's profile"""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    # Read file content
    file_content = await file.read()

    # Generate unique file path with user folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"avatars/{user['id']}_{timestamp}_{file.filename}"

    # Upload to Supabase storage
    try:
        image_url = supabase.upload_file(
            "images",
            file_path,
            file_content,
            file_options={"content-type": file.content_type},
        )

        # Update the user's profile with the image URL
        supabase.update_profile(user["id"], {"profile_image": image_url})

        return {
            "success": True,
            "image_url": image_url,
            "file_name": file.filename,
            "uploaded_at": timestamp,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


@router.put("")
async def update_profile(data: Dict[str, Any], user=Depends(get_current_user)):
    """Update the current user's profile with partial updates"""
    # Fetch the current profile
    current_profile = supabase.get_profile(user["id"])
    if not current_profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Update the profile with provided data
    updated_profile = {**current_profile, **data}

    # Check if all required fields are completed to mark as onboarded
    required_fields = [
        "allergies",
        "sports",
        "country",
        "sports_time_per_week",
        "diet_restrictions",
    ]
    is_onboarded = all(
        field in updated_profile and updated_profile[field] is not None
        for field in required_fields
    )

    if is_onboarded:
        updated_profile["is_onboarded"] = True

    return supabase.update_profile(user["id"], updated_profile)
