from langchain_core.messages import HumanMessage, SystemMessage
from models import llm_local

messages = [
    SystemMessage("Translate the following from English into Italian. Don't add any extra information."),
    HumanMessage("hi!"),
]

answer = llm_local.invoke(messages)
print(answer)

for token in llm_local.stream(messages):
    print(token.content, end="|")