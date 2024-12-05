from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph

from langgraph.graph.message import add_messages

from models import llm_local


# Define the state
class DialogueState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_speaker: str


# Create two different models for the philosophers
philosopher1_llm = llm_local
philosopher2_llm = llm_local


def create_philosopher(name: str, perspective: str, llm):
    prompt = f"""You are {name}, a philosopher with the following perspective: {perspective}
    Engage in a philosophical dialogue, building on the previous message and sharing your views.
    Keep responses thoughtful but concise (2-3 sentences)."""

    def philosopher_node(state: DialogueState):
        if state["current_speaker"] != name:
            return {}

        # Get the last message
        last_message = state["messages"][-1]

        # Generate response using the philosophical perspective
        response = llm.invoke([
            HumanMessage(content=prompt),
            HumanMessage(content=f"Previous message: {last_message.content}\nYour response:")
        ])

        print(f"{name} says: {response.content}")

        # Create a HumanMessage from the response for the other philosopher
        message = HumanMessage(content=f"{name} says: {response.content}")

        # Switch to other speaker
        next_speaker = "Philosopher2" if name == "Philosopher1" else "Philosopher1"

        return {
            "messages": [message],
            "current_speaker": next_speaker
        }

    return philosopher_node


# Create the graph
graph = StateGraph(DialogueState)

# Add philosopher nodes with different models
philosopher1 = create_philosopher(
    "Philosopher1",
    "You follow Aristotelian virtue ethics and believe in the golden mean",
    philosopher1_llm
)
philosopher2 = create_philosopher(
    "Philosopher2",
    "You are a Kantian deontologist focused on categorical imperatives",
    philosopher2_llm
)

# Add nodes and edges
graph.add_node("Philosopher1", philosopher1)
graph.add_node("Philosopher2", philosopher2)
graph.add_edge("Philosopher1", "Philosopher2")
graph.add_edge("Philosopher2", "Philosopher1")

# Set entry point
graph.set_entry_point("Philosopher1")

# Compile the graph
compiled_graph = graph.compile()

# Run the dialogue
initial_state = {
    "messages": [HumanMessage(content="What is the nature of moral behavior?")],
    "current_speaker": "Philosopher1"
}

# Run for a few turns
for _ in range(4):
    for event in compiled_graph.stream(initial_state):
        if "messages" in event:
            print(f"\n{event['current_speaker']} will respond next")
            print(event["messages"][-1].content)