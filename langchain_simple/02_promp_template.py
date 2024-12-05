from langchain_core.prompts import ChatPromptTemplate
from models import llm_local
system_template = "Translate the following from English into {language}. Don't add any extra information."

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)

response = llm_local.invoke(prompt)
print(response)