from fastapi import FastAPI

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangchainPydanticBaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

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
    - Include ALL meals of the day: breakfast, mid-morning snack, lunch, afternoon snack, and dinner (5 meals per day, if available).
"""


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=False)


class GenerateMealPlanRequest(PydanticBaseModel):
    pdf_url: str
    language: str


class GenerateRecipeBreakdownRequest(PydanticBaseModel):
    recipe_url: str
    language: str


class GenerateMealPlanIngredientsRequest(PydanticBaseModel):
    meal_plan: str


class GenerateMealPlanRecommendationsRequest(PydanticBaseModel):
    meal_plan: str


def get_recipe_breakdown_docs(url):
    loader = WebBaseLoader(url)
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


class RecipeBreakdown(LangchainPydanticBaseModel):
    title: str = Field(description="The title of the recipe")
    author: str = Field(description="The author of the recipe")
    difficulty: str = Field(description="The difficulty level of the recipe")
    time: str = Field(description="The time it takes to prepare the recipe")
    servings: str = Field(description="The number of servings the recipe makes")
    ingredients: list[str] = Field(description="The ingredients needed for the recipe")
    steps: list[str] = Field(description="The steps to prepare the recipe")


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
    response = chain.invoke(
        {
            "context": documents,
            "language": request_body.language,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return response


class Meal(LangchainPydanticBaseModel):
    mealType: str = Field(description="The type of meal")
    meal: str = Field(description="The meal details")
    ingredients: list[str] = Field(
        description="The ingredients needed for the meal without quantities"
    )


class MealPlan(LangchainPydanticBaseModel):
    day: str = Field(description="The day of the week")
    meals: list[Meal] = Field(description="The meals for the day")


class MealPlanResult(LangchainPydanticBaseModel):
    results: list[MealPlan] = Field(description="The generated meal plan")


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
    response = chain.invoke(
        {
            "context": documents,
            "language": request_body.language,
            "instructions": get_mealplan_prompt,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return response


# def generate_mealplan_ingredients(meal_plan):
#     messages = [
#         ("system", get_mealplan_ingredients_prompt),
#         (
#             "human",
#             meal_plan,
#         ),
#     ]

#     return llm.invoke(messages)


@app.post("/recipes/breakdown")
async def get_breakdown(request_body: GenerateRecipeBreakdownRequest):
    recipe_breakdown = get_recipe_breakdown(request_body)

    return recipe_breakdown


@app.post("/mealplans/generate/meals")
async def generate_mealplan(request_body: GenerateMealPlanRequest):
    meal_plan = generate_mealplan_from_pdf(request_body)

    return meal_plan


# @app.post("/mealplans/generate/ingredients")
# async def generate_mealplan(request_body: GenerateMealPlanIngredientsRequest):
#     ingredients = generate_mealplan_ingredients(request_body.meal_plan)

#     return ingredients
