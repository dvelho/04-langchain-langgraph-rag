#https://python.langchain.com/docs/tutorials/agents/
from typing import Optional, Type
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from models import llm, search


#Create a fake weather API

def get_current_weather(location, unit):
    # Call an external API to get relevant information (like serpapi, etc)
    # Here for the demo we will send a mock response
    weather_info = {
        "location": location,
        "temperature": "78",
        "unit": unit,
        "forecast": ["sunny", "with a chance of meatballs"],
    }
    return weather_info

class GetCurrentWeatherCheckInput(BaseModel):
    # Check the input for Weather
    location: str = Field(..., description = "The name of the location name for which we need to find the weather")
    unit: str = Field(..., description = "The unit for the temperature value")


class GetCurrentWeatherTool(BaseTool):
    name: str = "get_current_weather"
    description: str = "Used to find the weather for a given location in said unit"

    def _run(self, location: str, unit: str):
        # print("I am running!")
        weather_response = get_current_weather(location, unit)
        return weather_response

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = GetCurrentWeatherCheckInput

#create an agent
memory = MemorySaver()

weather_tool = GetCurrentWeatherTool()
tools = [weather_tool]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live, in Celsius?")]}, config
):
    print(chunk)
    print("----")