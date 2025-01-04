import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangchainPydanticBaseModel
from langchain_core.pydantic_v1 import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import Annotated, TypedDict

from src.utils.execution_timer import ExecutionTimer
from src.utils.pdf_utils import pdf_to_base64_images
from src.utils.supabase_client import supabase

load_dotenv()


app = FastAPI()


security = HTTPBearer()


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
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user's profile
        profile = supabase.get_profile(user.user.id)
        if not profile:
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
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


llm = ChatGoogleGenerativeAI(
    temperature=0,
    max_retries=2,
    model="gemini-2.0-flash-exp",
)


class Meal(TypedDict):
    slot: Annotated[
        int,
        ...,
        "The type of meal in slots, 0 for breakfast, 1 for mid morning snack, 2 for lunch, 3 for afternoon snack, 4 for dinner",
    ]
    meal: Annotated[str, ..., "The meal details including quantities and ingredients"]
    recipeQuery: Annotated[
        str,
        ...,
        "A simplified recipe search query. For example, if meal is '120g pasta, 200g chicken breast, vegetables', the query should be 'healthy pasta chicken vegetables recipe'",
    ]


class MealPlan(TypedDict):
    weekday: Annotated[
        int,
        ...,
        "The day of the week in weekday format (e.g. 0 for Monday, 1 for Tuesday, etc.)",
    ]
    meals: Annotated[list[Meal], ..., "The meals for the day"]


class MealPlanResult(TypedDict):
    results: Annotated[list[MealPlan], ..., "The generated meal plan"]


class RecipeBreakdown(LangchainPydanticBaseModel):
    thumbnail: str = Field(description="The thumbnail image of the recipe")
    title: str = Field(description="The title of the recipe")
    author: str = Field(description="The author of the recipe")
    difficulty: str = Field(description="The difficulty level of the recipe")
    time: str = Field(description="The time it takes to prepare the recipe")
    servings: str = Field(description="The number of servings the recipe makes")
    ingredients: list[str] = Field(description="The ingredients needed for the recipe")
    steps: list[str] = Field(description="The steps to prepare the recipe")


class GenerateMealPlanRequest(PydanticBaseModel):
    pdf_url: str
    language: str


class RecipeRequest(PydanticBaseModel):
    title: str
    thumbnail: Optional[str] = None
    author: Optional[str] = None
    difficulty: Optional[str] = None
    time: Optional[str] = None
    servings: Optional[str] = None
    ingredients: List[str]
    steps: List[str]


class MealRequest(PydanticBaseModel):
    slot: int
    meal: str
    ingredients: list[str]
    recipe: Optional[RecipeRequest] = None


class MealPlanRequest(PydanticBaseModel):
    weekday: int
    meals: list[MealRequest]
    pdf_url: Optional[str] = None
    data: Dict[str, Any]


class GenerateMealPlanRecommendationsRequest(PydanticBaseModel):
    meal_plan: list[MealPlanRequest]


class GenerateRecipeBreakdownRequest(PydanticBaseModel):
    recipe_url: str
    language: str


class DietaryPreferences(PydanticBaseModel):
    target_calories: Optional[int] = None
    dietary_restrictions: Optional[List[str]] = None
    cuisines: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class SignUpRequest(PydanticBaseModel):
    email: str
    password: str
    first_name: str
    last_name: str


class SignInRequest(PydanticBaseModel):
    email: str
    password: str


def get_recipe_breakdown_docs(url):
    loader = WebBaseLoader(url)
    # loader = AsyncChromiumLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)

    return documents


@ExecutionTimer.time_this
def get_mealplan_docs(url: str):
    loader = PyPDFLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)

    return documents


def get_recipe_breakdown(request_body: GenerateRecipeBreakdownRequest):
    documents = get_recipe_breakdown_docs(request_body.recipe_url)
    parser = JsonOutputParser(pydantic_object=RecipeBreakdown)
    prompt = PromptTemplate.from_template(
        """
            Summarize the recipe with the following information:
            Answer must be written in: {language}

            format_instructions: {format_instructions}

            {context}
        """
    )
    chain = create_stuff_documents_chain(
        llm,
        prompt,
        output_parser=parser,
    )
    with get_openai_callback() as cb:
        response = chain.invoke(
            {
                "context": documents,
                "language": request_body.language,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        print(cb)

    return response


@ExecutionTimer.time_this
async def generate_mealplan_from_pdf(
    file_content: bytes, language: str, preferences: Optional[DietaryPreferences] = None
) -> List[MealPlanResult]:
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

        full = None

        async for chunk in structured_llm.astream(messages):
            print(chunk, flush=True)
            full = chunk

        return full

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def generate_mealplan_recommendations(
    request_body: GenerateMealPlanRecommendationsRequest,
):
    now = datetime.now()
    current_weekday = now.weekday()

    # Search
    google_search = GoogleSerperAPIWrapper()

    current_day_mealplan_list = [
        {
            "weekday": mealplan.weekday,
            "meals": [
                {"slot": meal.slot, "meal": meal.meal, "ingredients": meal.ingredients}
                for meal in mealplan.meals
            ],
        }
        for mealplan in filter(
            lambda meal: meal.weekday == current_weekday, request_body.meal_plan
        )
    ]

    google_search = GoogleSerperAPIWrapper(type="images")

    recommendations = []

    for mealplan in current_day_mealplan_list:
        for meal in mealplan["meals"]:
            search_result = google_search.results(f"{meal['meal']} recipe")
            top_3_results = search_result["images"][:3]

            for result in top_3_results:
                recommendation = {
                    "recipe_title": result["title"],
                    "recipe_link": result["link"],
                    "recipe_thumbnail": result["imageUrl"],
                    "weekday": mealplan["weekday"],
                    "slot": meal["slot"],
                }
                recommendations.append(recommendation)

    return recommendations


# Auth endpoints
@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    """Sign up a new user"""
    return await supabase.sign_up(
        request.email, request.password, request.first_name, request.last_name
    )


@app.post("/auth/signin")
async def signin(request: SignInRequest):
    """Sign in a user"""
    return await supabase.sign_in(request.email, request.password)


@app.post("/auth/signout")
async def signout(user=Depends(get_current_user)):
    """Sign out the current user"""
    await supabase.sign_out()
    return {"message": "Successfully signed out"}


@app.get("/profile")
async def get_profile(user=Depends(get_current_user)):
    """Get the current user's profile"""
    profile = supabase.get_profile(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.post("/profile/image")
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


@app.put("/profile")
async def update_profile(data: Dict[str, Any], user=Depends(get_current_user)):
    """Update the current user's profile"""
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


@app.post("/recipes/breakdown")
async def get_breakdown(
    request_body: GenerateRecipeBreakdownRequest, user=Depends(get_current_user)
):
    recipe_breakdown = get_recipe_breakdown(request_body)
    return recipe_breakdown


@app.post("/mealplans/generate/recommendations")
async def generate_mealplan_recommendations_endpoint(
    user=Depends(get_current_user),
):
    try:
        # Fetch the latest meal plan for the user
        meal_plans = supabase.list_meal_plans(user["id"])
        if not meal_plans:
            raise HTTPException(
                status_code=404, detail="No meal plans found for the user."
            )

        latest_meal_plan = meal_plans[0]  # Assuming the latest is the last in the list

        google_search = GoogleSerperAPIWrapper(type="images")
        recommendations = []

        for meal in latest_meal_plan["meals"]:
            # Use the existing recipeQuery for searching recipes
            search_result = google_search.results(meal["recipeQuery"])
            top_3_results = search_result["images"][:3]

            for result in top_3_results:
                recommendation = {
                    "recipe_title": result["title"],
                    "recipe_link": result["link"],
                    "recipe_thumbnail": result["imageUrl"],
                    "weekday": meal["weekday"],
                    "slot": meal["slot"],
                }
                recommendations.append(recommendation)

        # Save recommendations to the database
        supabase.save_recommendations(
            user["id"], latest_meal_plan["id"], recommendations
        )

        return {"success": True, "recommendations": recommendations}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


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
            if full_response is None:
                full_response = chunk
            else:
                full_response = chunk

            yield f"data: {json.dumps({'type': 'update', 'content': chunk})}\n\n"

        # Send the final complete response
        yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


async def save_generated_meal_plan(
    user_id: str, pdf_url: str, meal_plan_data: Dict[str, Any]
):
    """Save the generated meal plan to the database"""
    try:
        # Create meal plan
        meal_plan_request = {
            "pdf_url": pdf_url,
            "data": meal_plan_data,
        }

        supabase.create_meal_plan(user_id, meal_plan_request)
    except Exception as e:
        print(f"Error saving meal plan: {e}")
        # Don't raise the exception - we want to continue returning the streaming response
        # but log the error for debugging


@app.post("/mealplans/generate/overview")
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


# Endpoint to check if the current user has uploaded a meal plan
@app.get("/mealplans/exists")
async def check_meal_plan_exists(user=Depends(get_current_user)):
    """Check if the current user has uploaded a meal plan"""
    files = await supabase.list_files("pdfs", path=f"meal_plans/{user['id']}")
    return {"exists": len(files) > 0}


@app.post("/mealplans")
async def create_meal_plan(meal_plan: MealPlanRequest, user=Depends(get_current_user)):
    """Create a new meal plan with optional recipes"""
    return await supabase.create_meal_plan(user["id"], meal_plan.dict())


@app.get("/mealplans/{meal_plan_id}/with-recipes")
async def get_meal_plan_with_recipes(meal_plan_id: str, user=Depends(get_current_user)):
    """Get a meal plan with all its recipes"""
    meal_plan = await supabase.get_meal_plan_with_recipes(meal_plan_id)
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return meal_plan


@app.get("/mealplans/{meal_plan_id}")
async def get_meal_plan(meal_plan_id: str, user=Depends(get_current_user)):
    """Get a meal plan by ID"""
    meal_plan = await supabase.get_meal_plan(meal_plan_id)
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return meal_plan


@app.get("/mealplans")
async def list_meal_plans(user=Depends(get_current_user)):
    """List all meal plans"""
    return supabase.list_meal_plans()


@app.put("/mealplans/{meal_plan_id}")
async def update_meal_plan(
    meal_plan_id: str, meal_plan: MealPlanRequest, user=Depends(get_current_user)
):
    """Update a meal plan"""
    return await supabase.update_meal_plan(meal_plan_id, meal_plan.dict())


@app.delete("/mealplans/{meal_plan_id}")
async def delete_meal_plan(meal_plan_id: str, user=Depends(get_current_user)):
    """Delete a meal plan"""
    await supabase.delete_meal_plan(meal_plan_id)
    return {"message": "Meal plan deleted"}


# Recipes CRUD endpoints
@app.post("/recipes")
async def create_recipe(
    recipe: GenerateRecipeBreakdownRequest, user=Depends(get_current_user)
):
    """Create a new recipe"""
    recipe_data = get_recipe_breakdown(recipe)
    return await supabase.create_recipe(user["id"], recipe_data)


@app.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: str, user=Depends(get_current_user)):
    """Get a recipe by ID"""
    recipe = await supabase.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


@app.get("/recipes")
async def list_recipes(user=Depends(get_current_user)):
    """List all recipes"""
    return await supabase.list_recipes()


@app.put("/recipes/{recipe_id}")
async def update_recipe(
    recipe_id: str,
    recipe: GenerateRecipeBreakdownRequest,
    user=Depends(get_current_user),
):
    """Update a recipe"""
    recipe_data = get_recipe_breakdown(recipe)
    return await supabase.update_recipe(recipe_id, recipe_data)


@app.delete("/recipes/{recipe_id}")
async def delete_recipe(recipe_id: str, user=Depends(get_current_user)):
    """Delete a recipe"""
    await supabase.delete_recipe(recipe_id)
    return {"message": "Recipe deleted"}


@app.post("/mealplans/upload")
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


@app.get("/mealplans/upload/latest")
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
