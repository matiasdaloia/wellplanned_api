from fastapi import FastAPI

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

query = """
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


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=False)


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_context_retriever_chain(vector_store):

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input: str, website_url: str):
    retriever_chain = get_context_retriever_chain(get_vectorstore_from_url(website_url))
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({"input": user_input})

    return response["answer"]


@app.get("/playwright")
async def read_item():
    response = get_response(
        query,
        "https://ekilu.com/es/receta/pasta-integral-con-pollo-y-verduras",
    )

    return response
