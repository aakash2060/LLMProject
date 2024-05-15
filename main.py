from flask import Flask, request, render_template, session
import os
import socket
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin


# Set OpenAI API key
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'secretKey'  # Set your secret key for session management

# Global variables
document_search = None
chain = None

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

@app.route("/", methods=['GET','POST'])
@cross_origin()
def hello():
    global document_search, chain

    answers = session.get('answers', [])
    if answers is None:
        answers = []

    if request.method == 'POST':
        if 'file' in request.files:
            # If file is uploaded, save it and update the model
            f = request.files['file']
            if f.filename != '':
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                f.save(pdf_path)
                update_model(pdf_path)

        if "query" in request.form: 
            query = request.form['query']
            if document_search is not None and chain is not None and query is not None:
                docs = document_search.similarity_search(query)
                answer = chain.run(input_documents=docs, question=query)
                question_answer = {query: answer}
                answers.append(question_answer)
                session['answers'] = answers
                print('THE ANSWERS ARE ', answers)
        #  
        # print('THE QUERY IS ', query)
        # # Perform similarity search and run the question answering chain
      

    return render_template('form.html', name=os.getenv("NAME", "world"), hostname=socket.gethostname(), answers=answers)

def update_model(pdf_path):
    global document_search, chain

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

if __name__ == "__main__":
    # Define the upload folder
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=8000)
