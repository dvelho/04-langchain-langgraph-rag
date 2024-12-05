#loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from models import embed_model, llm_local

file_path = "../docs/EMPLOYEE_PRIVACY_POLICY_MINDERA_PT.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

#split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)



#vectorstore
from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=embed_model)

#add documents
ids = vector_store.add_documents(documents=all_splits)

#results = vector_store.similarity_search(
#    "What is this policy about?"
#)

#print(results[0])


#retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


#chat

# helper Combine the documents into a single string
def format_docs(docs):
    return "".join(d.page_content for d in docs)

#Define a system prompt that tells the model how to use the retrieved context
system_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""



prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{question}")]
)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm_local
    | StrOutputParser()
)

response = chain.invoke("What is this policy about?")
print(response)