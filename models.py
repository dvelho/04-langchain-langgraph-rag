from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
embed_model = OpenAIEmbeddings(check_embedding_ctx_length=False, base_url="http://localhost:1234/v1", api_key="key", model="text-embedding-nomic-embed-text-v1.5@q8_0")
llm_local = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="key", model="llama-3.2-3b-instruct", max_tokens=100)
llm = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash')
search = TavilySearchResults(max_results=2)

