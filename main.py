import json
import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
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

load_dotenv()


app = FastAPI()


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
        "A Google friendly query to search for related recipes for the meal",
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


class MealRequest(PydanticBaseModel):
    slot: int
    meal: str
    ingredients: list[str]


class MealPlanRequest(PydanticBaseModel):
    weekday: int
    meals: list[MealRequest]


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


@app.post("/recipes/breakdown")
async def get_breakdown(request_body: GenerateRecipeBreakdownRequest):
    recipe_breakdown = get_recipe_breakdown(request_body)

    return recipe_breakdown


@app.post("/mealplans/generate/recommendations")
async def generate_mealplan(request_body: GenerateMealPlanRequest):
    meal_plan = generate_mealplan_recommendations(request_body)

    return meal_plan


async def stream_mealplan(
    file_content: bytes, language: str, preferences: Optional[DietaryPreferences] = None
):
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


@app.post("/mealplans/generate/overview")
async def generate_mealplan(
    file: UploadFile = File(...),
    language: str = Form("en"),
    preferences: Optional[str] = Form(None),
):
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_content = await file.read()

    if len(file_content) > 32 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 32MB limit")

    return StreamingResponse(
        stream_mealplan(
            file_content=file_content, language=language, preferences=preferences
        ),
        media_type="text/event-stream",
    )
