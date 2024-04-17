from flask import Flask, request, render_template,session
# from redis import Redis, RedisError
import os
import socket

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Set OpenAI API key
load_dotenv()

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Provide the path of the PDF file
pdf_path = '1210 HOOKS 4.pdf'

# Read text from the PDF file
pdf_reader = PdfReader(pdf_path)
raw_text = ''
for page in pdf_reader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Split the text using Character Text Splitter to manage token size
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Create a FAISS index from the texts
document_search = FAISS.from_texts(texts, embeddings)

# Load the question answering chain using OpenAI model
chain = load_qa_chain(OpenAI(), chain_type="stuff")

app = Flask('234')
app.secret_key = 'secretKey'  # Set your secret key for session management



@app.route("/", methods=['GET', 'POST'])
def hello():
    answers = session.get('answers')
    if answers ==  None :
        answers = []

    if request.method == 'POST':
        query = request.form['query']
        print('THE QUERY IS ',query)
        # Perform similarity search and run the question answering chain
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        question_answer = {query: answer}
        
        
        answers.append(question_answer)
        session['answers'] = answers

        print('THE ANSWERS ARE ',answers)

    return render_template('form.html', name= os.getenv("NAME", "world"), hostname=socket.gethostname(), answers=answers)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
