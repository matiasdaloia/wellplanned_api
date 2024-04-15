from fastapi import FastAPI

from typing import Union

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


app = FastAPI()

get_recipe_breakdown_prompt = """
    Given the following recipe text, generate a detailed summary in JSON format. Make sure to include ALL ingredients and step-by-step preparation instructions. You must not omit any steps. The summary should be clear and easy to follow. Do not include any additional information that is not relevant to the preparation of the recipe.

    Example of format:
    {
        recipe: {
            title: 'Pasta Integral con Pollo y Verduras',
            author: 'Equipo ekilu',
            difficulty: 'Baja',
            time: '20 minutos',
            servings: '2 Porciones',
            ingredients: {
            fresh: [
                '1 pechuga de pollo',
                '1/2 cebolla',
                '1 diente de ajo',
                '1/2 calabacín',
                '8 tomates cherry'
            ],
            pantry: [
                '180 gramos de pasta integral',
                '2 cucharadas de aceite de oliva',
                'Sal',
                'Pimienta negra'
            ]
            },
            steps: [
                'En una olla con agua hirviendo, cocina la pasta según las instrucciones del envase. Escurre y reserva.',
                'Trocea la pechuga de pollo, cebolla, ajo, calabacín y tomates.',
                'En una sartén con aceite caliente, saltea el pollo con sal y pimienta. Retira y reserva.',
                'En la misma sartén, añade un poco más de aceite y cocina las verduras troceadas (excepto los tomates) a fuego medio por 7-8 minutos.',
                'Agrega los tomates y el pollo reservado a la sartén. Saltea todo durante 4-5 minutos.',
                'Incorpora la pasta cocida a la sartén, mezcla y deja integrar sabores por un par de minutos.',
                'Sirve en dos platos y opcionalmente, espolvorea queso rallado por encima.'
            ]
        }
    }
"""

get_mealplan_prompt = """
    You will be provided with a patient diet plan from a nutritionist. Your task is to generate a 7-day meal plan for the patient following these guidelines:
    - For each meal, only select one of the available options, not all of them.
    - Include quantity of the ingredients for each meal if available.
    - When there are multiple protein or main dish options, choose only one.
    - Include ALL meals of the day: breakfast, mid-morning snack, lunch, afternoon snack, and dinner (5 meals per day, if available).
    - Result must not be translated, only provide the information in the same language as the input.
    - Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation:

    {
        "results": [
            {
            "day": "{{ day of the week }}",
            "meals": [
                {
                "mealType":
                    "{{ breakfast | midMorningSnack | lunch | afternoonSnack | dinner }}",
                "meal": "{{ detailed meal info with quantities and ingredients }}",
                },
            ],
            },
        ],
    }
"""

get_mealplan_ingredients_prompt = """
     You will be provided with a patient diet plan from a nutritionist with the following JSON structure:

     Your task is to generate:
     1) A detailed list of ingredients for each lunch and dinner meals, in an array of strings.
     2) A grocery list of ingredients including all day meals, not only lunch and dinner. If an ingredient is repeated in different meals, you should sum the quantities needed for each meal.

     Follow these guidelines:
     - In grocery list, include ingredients for other meals that are not included in the detailed list (breakfast, mid-morning snack, afternoon snack, etc.)
     - Do not include additional information or text.
     - Result must not be translated, only provide the information in the same language as the input.
     - Valid units are: COUNT, CLOVES, SLICES, STALKS, LEAVES, BUNCHES, KILOGRAMS, GRAMS, POUNDS, OUNCES, PINCHES, LITERS, CENTILITERS, MILLILITERS, CC, DROPS, GALLONS, QUARTS, PINTS, CUPS, FL_OZ, HEAPING_TBSP, TBSP, HEAPING_TSP, TSP
     - Units must be written as in the previous guideline, if the unit is in its minified form, write it in full (e.g. write MILLILITERS instead of ML)
     - Remove any double quotes that can affect JSON format, or just replace it with &apos;
     - Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation:


     {
        ingredientsForRecipes: [
            {
            day: "{{ day of the week }}",
            dinner: [
                "{{ ingredient name }} | {{ quantity needed for the meal without unit }} | {{ unit of measurement for the ingredient (should be only measured in KILOGRAMS or COUNT) }}",
            ],
            lunch: [
                "{{ ingredient name }} | {{ quantity needed for the meal without unit }} | {{ unit of measurement for the ingredient (should be only measured in KILOGRAMS or COUNT) }}",
            ],
            },
        ],
        groceryList: [
            "{{ ingredient name }} | {{ quantity needed for the week without unit }} | {{ unit of measurement for the ingredient (should be only measured in KILOGRAMS or COUNT) }}",
        ],
     }
"""

prompt_template = """
    {context}

    Question: {question}

    Helpful Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=False)


def get_recipe_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_mealplan_vectorstore_from_url(url: str):
    loader = PyPDFLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_recipe_breakdown(website_url: str):
    vector_store = get_recipe_vectorstore_from_url(website_url)
    rag_chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(get_recipe_breakdown_prompt)

    vector_store.delete_collection()

    return response


def generate_mealplan_from_pdf(pdf_url: str):
    vector_store = get_mealplan_vectorstore_from_url(pdf_url)
    rag_chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(get_mealplan_prompt)

    vector_store.delete_collection()

    return response


def generate_mealplan_ingredients(meal_plan):
    messages = [
        ("system", get_mealplan_ingredients_prompt),
        (
            "human",
            meal_plan,
        ),
    ]

    return llm.invoke(messages)


class GenerateMealPlanRequest(BaseModel):
    pdf_url: str


class GenerateMealPlanIngredientsRequest(BaseModel):
    meal_plan: str


class GenerateMealPlanRecommendationsRequest(BaseModel):
    meal_plan: str


@app.get("/recipes/breakdown")
async def get_breakdown(recipe_url: Union[str, None] = None):
    recipe_breakdown = get_recipe_breakdown(recipe_url)

    return recipe_breakdown


@app.post("/mealplans/generate/meals")
async def generate_mealplan(request_body: GenerateMealPlanRequest):
    meal_plan = generate_mealplan_from_pdf(request_body.pdf_url)

    return meal_plan


@app.post("/mealplans/generate/ingredients")
async def generate_mealplan(request_body: GenerateMealPlanIngredientsRequest):
    ingredients = generate_mealplan_ingredients(request_body.meal_plan)

    return ingredients
