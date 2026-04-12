from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

pdf_path = Path(__file__).parent / "Dsa_Question.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key= os.getenv("OPENAI_API_KEY"),
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333/",
#     collection_name="learning_langchain",
#     embedding=embedder
# )
# vector_store.add_documents(documents=split_docs)

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333/",
    collection_name="learning_langchain",
    embedding=embedder
)

relevent_chunks = retriver.similarity_search(
    query="Armstrong Number using java"
)

system_prompts =""""
You are an AI assistent who responds base of the available context.

context:
{relevent_chunks}
"""


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

while True:
    user_input = input(">")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":system_prompts},
            {"role":"user","content":user_input}
        ]
    )

    parsed_output = json.loads(response.choices[0].message.content)

    print(parsed_output.get("content"))