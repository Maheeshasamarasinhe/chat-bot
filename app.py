from flask import Flask, render_template, jsonify, request, make_response
from flask_cors import CORS
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def set_headers(response):
    # Allow embedding in iframe from any origin
    if 'X-Frame-Options' in response.headers:
        del response.headers['X-Frame-Options']
    response.headers['Content-Security-Policy'] = "frame-ancestors *"
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embedding = download_embeddings()

index_name = "maternal-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
# Load Existing index 

from langchain_pinecone import PineconeVectorStore
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3  # Low temperature = More factual answers (Good for Medical)
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Re-create the chain with the new prompt

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8081, debug= True)