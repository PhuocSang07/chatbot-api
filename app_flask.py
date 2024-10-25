from langchain_mongodb import MongoDBAtlasVectorSearch
from model import LanguageModelPipeline
from pymongo import MongoClient
from dotenv import load_dotenv
import gradio as gr
import torch
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import json
import time

app = Flask(__name__)
CORS(app)


#---- Enviroment Variables ----#
load_dotenv()
model_embedding_name = os.getenv('MODEL_EMBEDDING_NAME') or 'bkai-foundation-models/vietnamese-bi-encoder'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---- MongoDB Atlas ----#
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')
DB_NAME = os.getenv('DB_NAME') or 'langchain_db'
COLLECTION_NAME = os.getenv('COLLECTION_NAME') or 'vector_db'
ATLAS_VECTOR_SEARCH_INDEX_NAME  = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME') or 'vector_search_index'

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[DB_NAME][COLLECTION_NAME]
print("Kết nối thành công:", client[DB_NAME].name)

#---- Language Model ----#
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             "Bạn là một trợ lí Calendar, bạn có nhiệm vụ trích xuất các thông tin từ sự kiện được cung cấp. Các thông tin cần trích xuất bao gồm summary, location, description, start (dateTime, timeZone), end (dateTime, timeZone), recurrence, attendees, reminders(useDefault, overrides(method, minutes)). Trình bày thông tin trích xuất được theo định dạng JSON."
        ),
        ("human", "{question}"),
    ]
)

pipeline = LanguageModelPipeline(model_embedding_name=model_embedding_name)
embedding = pipeline.get_embedding()

#---- MongoDB Atlas Vector Search ----#
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 15})
llm_chain = pipeline.create_chain_reranking(llm, prompt, retriever, True)

#---- Route ----#
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('message')
        
        if question:
            start = time.time()
            result = llm_chain(question)['result']
            end = time.time()

            return jsonify({
                'question': question,
                'response': str(result),
                'time': end - start
            })
        else:
            return jsonify({'error': 'No question provided'}), 400
    
    except Exception as e:
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
