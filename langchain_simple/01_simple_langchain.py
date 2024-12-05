from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from models import llm_local

messages = [
    SystemMessage("Translate the following from English into Italian. Don't add any extra information. Just translate the text."),
    HumanMessage("hi!"),
    AIMessage("Ciao!"),
    HumanMessage("How are you?"),
    AIMessage("Come stai?"),
    HumanMessage("I'm fine, thank you."),
]

#answer = llm_local.invoke(messages)
#print(answer)

#Add a parser
#parser = StrOutputParser()
#answer = parser.parse(answer.content)
#print(answer)

for token in llm_local.stream(messages):
    print(token.content, end='')


