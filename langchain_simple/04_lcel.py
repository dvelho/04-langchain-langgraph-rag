from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models import llm_local

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | llm_local | StrOutputParser()

response = chain.stream({"topic": "elephants, cats and planets"})

for token in response:
    print(token, end='')

