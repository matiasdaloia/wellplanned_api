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
    - The result must be in the language of the provided meal plan.
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
    retriever = get_recipe_vectorstore_from_url(website_url).as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(get_recipe_breakdown_prompt)

    return response


def generate_mealplan_from_pdf(pdf_url: str):
    retriever = get_mealplan_vectorstore_from_url(pdf_url).as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(get_mealplan_prompt)

    return response


@app.get("/recipes/breakdown")
async def get_breakdown(q: Union[str, None] = None):
    if q is None:
        return {"message": "Please provide a URL"}
    else:
        response = get_recipe_breakdown(q)

    return response


@app.post("/mealplans/generate")
async def generate_mealplan(q: Union[str, None] = None):
    if q is None:
        return {"message": "Please provide a URL"}
    else:
        response = generate_mealplan_from_pdf(q)

    return response
