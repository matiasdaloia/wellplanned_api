from fastapi import FastAPI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangchainPydanticBaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_loaders import WebBaseLoader


from langchain_community.utilities import GoogleSerperAPIWrapper

from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader

from pydantic import BaseModel as PydanticBaseModel

from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()


get_mealplan_prompt = """
    You will be provided with a patient diet plan from a nutritionist. Your task is to generate a 7-day meal plan for the patient following these guidelines:
    - For each meal, only select one of the available options, not all of them.
    - Include quantity of the ingredients for each meal if available.
    - When there are multiple protein or main dish options, choose only one.
    - Include ALL meals of the day
"""


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=False)


class Meal(LangchainPydanticBaseModel):
    slot: int = Field(
        description="The type of meal in slots, 0 for breakfast, 1 for mid morning snack, 2 for lunch, 3 for afternoon snack, 4 for dinner"
    )
    meal: str = Field(description="The meal details")
    ingredients: list[str] = Field(
        description="The ingredients needed for the meal without quantities"
    )


class MealPlan(LangchainPydanticBaseModel):
    weekday: int = Field(
        description="The day of the week in weekday format (e.g. 0 for Monday, 1 for Tuesday, etc.)"
    )
    meals: list[Meal] = Field(description="The meals for the day")


class MealPlanResult(LangchainPydanticBaseModel):
    results: list[MealPlan] = Field(description="The generated meal plan")


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


def get_recipe_breakdown_docs(url):
    loader = WebBaseLoader(url)
    # loader = AsyncChromiumLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)

    return documents


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


def generate_mealplan_from_pdf(request_body: GenerateMealPlanRequest):
    documents = get_mealplan_docs(request_body.pdf_url)
    parser = JsonOutputParser(pydantic_object=MealPlanResult)
    prompt = PromptTemplate.from_template(
        """
            Instructions: {instructions}
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
                "instructions": get_mealplan_prompt,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        print(cb)

    return response


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


@app.post("/mealplans/generate/overview")
async def generate_mealplan(request_body: GenerateMealPlanRequest):
    meal_plan = generate_mealplan_from_pdf(request_body)

    return meal_plan


@app.post("/mealplans/generate/recommendations")
async def generate_mealplan(request_body: GenerateMealPlanRecommendationsRequest):
    meal_plan = generate_mealplan_recommendations(request_body)

    return meal_plan
