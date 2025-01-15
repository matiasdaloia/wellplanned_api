import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.routers.auth import get_current_user
from src.utils.logger import get_logger, log_execution_time_async
from src.utils.pdf_utils import pdf_to_base64_images
from src.utils.supabase_client import supabase

logger = get_logger()
router = APIRouter(prefix="/mealplans", tags=["meal_plans"])


class Meal(TypedDict):
    slot: int
    meal: str
    recipeQuery: str


class MealPlan(TypedDict):
    weekday: int
    meals: List[Meal]


class MealPlanResult(TypedDict):
    results: List[MealPlan]


class RecipeRequest(BaseModel):
    title: str
    thumbnail: Optional[str] = None
    author: Optional[str] = None
    difficulty: Optional[str] = None
    time: Optional[str] = None
    servings: Optional[str] = None
    ingredients: List[str]
    steps: List[str]


class MealRequest(BaseModel):
    slot: int
    meal: str
    ingredients: list[str]
    recipe: Optional[RecipeRequest] = None


class MealPlanRequest(BaseModel):
    weekday: int
    meals: list[MealRequest]
    pdf_url: Optional[str] = None
    data: Dict[str, Any]


@log_execution_time_async
async def stream_mealplan(file_content: bytes):
    try:
        base64_images = pdf_to_base64_images(file_content)
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            stream_usage=True,
        )
        structured_llm = llm.with_structured_output(MealPlanResult)

        messages = [
            SystemMessage(
                content="""
            You will be provided with a patient diet plan from a nutritionist. Your task is to extract a 7-day meal plan for the patient following these guidelines:
                - For each meal, only select one of the available options, not all of them.
                - Include quantity of the ingredients for each meal if available.
                - When there are multiple protein or main dish options, choose only one.
                - Include ALL meals of the day
                - Include ALL days of the week
                - If it says "free choice", choose any meal of the same type (e.g. breakfast, lunch, etc.)
                - For the recipeQuery field, create a simplified search query. For example:
                  * If meal is "120g pasta, 200g chicken breast, vegetables" → recipeQuery should be "healthy pasta chicken vegetables recipe"
                  * If meal is "2 eggs, 30g cheese, bread" → recipeQuery should be "healthy eggs cheese toast recipe"
                  * Remove quantities and keep only main ingredients for better search results
            """
            )
        ]

        for base64_image in base64_images:
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    ]
                )
            )

        full_response = None
        async for chunk in structured_llm.astream(messages):
            logger.debug(
                "Received chunk from LLM", extra={"extra_data": {"chunk": str(chunk)}}
            )
            full_response = chunk
            yield f"data: {json.dumps({'type': 'update', 'content': chunk})}\n\n"

        # Send the final complete response
        yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@log_execution_time_async
async def save_generated_meal_plan(
    user_id: str, pdf_url: str, meal_plan_data: Dict[str, Any]
):
    """Save the generated meal plan to the database"""
    try:
        logger.info(
            "Saving meal plan",
            extra={"extra_data": {"user_id": user_id, "pdf_url": pdf_url}},
        )

        meal_plan_request = {
            "pdf_url": pdf_url,
            "data": meal_plan_data,
        }

        supabase.create_meal_plan(user_id, meal_plan_request)
        logger.info("Successfully saved meal plan")

    except Exception as e:
        logger.error("Error saving meal plan", exc_info=True)


@router.post("/generate/overview")
async def generate_mealplan(
    user=Depends(get_current_user),
):
    try:
        # Get the latest PDF URL and details
        files = await supabase.list_files("pdfs", path=f"meal_plans/{user['id']}")

        if not files:
            raise HTTPException(
                status_code=404,
                detail="No PDF files found. Please upload a meal plan PDF first.",
            )

        # Get latest file
        latest_file = sorted(files, key=lambda x: x["name"], reverse=True)[0]
        pdf_url = await supabase.get_file_url(
            "pdfs", f"meal_plans/{user['id']}/{latest_file['name']}"
        )

        # Download the file content
        file_content = await supabase.download_file(
            "pdfs", f"meal_plans/{user['id']}/{latest_file['name']}"
        )

        if len(file_content) > 32 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 32MB limit")

        # Create an async generator that will both stream the response and save the meal plan
        async def generate_and_save():
            full_response = None
            async for chunk in stream_mealplan(
                file_content=file_content,
            ):
                # Parse the chunk to get the meal plan data
                chunk_data = json.loads(chunk.replace("data: ", ""))

                # If this is the complete response, save the meal plan
                if chunk_data["type"] == "complete":
                    full_response = chunk_data["content"]
                    # Save the meal plan in the background
                    asyncio.create_task(
                        save_generated_meal_plan(user["id"], pdf_url, full_response)
                    )

                yield chunk

        return StreamingResponse(
            generate_and_save(),
            media_type="text/event-stream",
        )

    except Exception as e:
        print(f"Error generating meal plan: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating meal plan: {str(e)}"
        )


@router.get("/exists")
async def check_meal_plan_exists(user=Depends(get_current_user)):
    """Check if the current user has uploaded a meal plan"""
    files = await supabase.list_files("pdfs", path=f"meal_plans/{user['id']}")
    return {"exists": len(files) > 0}


@router.post("")
async def create_meal_plan(meal_plan: MealPlanRequest, user=Depends(get_current_user)):
    """Create a new meal plan with optional recipes"""
    return await supabase.create_meal_plan(user["id"], meal_plan.dict())


@router.get("/{meal_plan_id}/with-recipes")
async def get_meal_plan_with_recipes(meal_plan_id: str, user=Depends(get_current_user)):
    """Get a meal plan with all its recipes"""
    meal_plan = await supabase.get_meal_plan_with_recipes(meal_plan_id)
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return meal_plan


@router.get("/{meal_plan_id}")
async def get_meal_plan(meal_plan_id: str, user=Depends(get_current_user)):
    """Get a meal plan by ID"""
    meal_plan = await supabase.get_meal_plan(meal_plan_id)
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return meal_plan


@router.get("")
async def list_meal_plans(user=Depends(get_current_user)):
    """List all meal plans"""
    return supabase.list_meal_plans()


@router.put("/{meal_plan_id}")
async def update_meal_plan(
    meal_plan_id: str, meal_plan: MealPlanRequest, user=Depends(get_current_user)
):
    """Update a meal plan"""
    return await supabase.update_meal_plan(meal_plan_id, meal_plan.dict())


@router.delete("/{meal_plan_id}")
async def delete_meal_plan(meal_plan_id: str, user=Depends(get_current_user)):
    """Delete a meal plan"""
    await supabase.delete_meal_plan(meal_plan_id)
    return {"message": "Meal plan deleted"}


@router.post("/upload")
async def upload_mealplan_pdf(
    file: UploadFile = File(...), user=Depends(get_current_user)
):
    """Upload a PDF file for meal plan generation and return the storage URL"""

    # Validate file type
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    file_content = await file.read()

    # Check file size (32MB limit)
    if len(file_content) > 32 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 32MB limit")

    try:
        # Generate unique file path with user folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"meal_plans/{user['id']}/{timestamp}_{file.filename}"

        # Upload to Supabase storage
        pdf_url = await supabase.upload_file(
            "pdfs",
            file_path,
            file_content,
            file_options={"content-type": "application/pdf"},
        )

        return {
            "success": True,
            "pdf_url": pdf_url,
            "file_name": file.filename,
            "uploaded_at": timestamp,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/upload/latest")
async def get_latest_mealplan_pdf(user=Depends(get_current_user)):
    """Get the URL of the latest uploaded meal plan PDF for the current user"""
    try:
        # List files in user's directory
        files = await supabase.list_files("pdfs", path=f"meal_plans/{user['id']}")

        if not files:
            raise HTTPException(
                status_code=404, detail="No PDF files found for this user"
            )

        # Sort files by name (which includes timestamp) to get the latest
        latest_file = sorted(files, key=lambda x: x["name"], reverse=True)[0]

        # Get the URL for the latest file
        pdf_url = await supabase.get_file_url("pdfs", latest_file["name"])

        return {
            "success": True,
            "pdf_url": pdf_url,
            "file_name": latest_file["name"].split("_", 2)[
                -1
            ],  # Extract original filename
            "uploaded_at": latest_file["created_at"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving latest PDF: {str(e)}"
        )
