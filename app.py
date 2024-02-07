from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="qwen:1.8b", callbacks=[StreamingStdOutCallbackHandler()])

# Define the question to be answered
question = "What is this book about?"
#question = input("Your question here: ")

# Initialize the directory loader
raw_documents = DirectoryLoader('PDFdocs', 
                                glob="**/*.pdf", 
                                loader_cls=PyPDFLoader, 
                                show_progress=True, 
                                use_multithreading=True).load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# Load the embeddings into Chroma
print("Loading documents into Chroma\n")
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                        model_kwargs={'device': 'cpu'})

db = Chroma.from_documents(documents, embedding=embeddings)

print(f">>>Question: {question}\n")

prompt_template = """
### Instruction:
You're question answering AI assistant, who answers questions based upon provided research in a distinct and clear way.
Answers must be based only on the information from research.

## Research:
{context}

## Question:
{question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={'prompt': PROMPT}
)

qa_chain({"query": question})
