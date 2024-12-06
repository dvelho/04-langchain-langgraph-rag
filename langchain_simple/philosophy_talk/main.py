from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph

from langgraph.graph.message import add_messages

from models import llm_local


# Create two different models for the philosophers
philosopher1_llm = llm_local
philosopher2_llm = llm_local

class DialogueState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_speaker: str
    summary: str
    last_summarized_idx: int  # New field to track last summarized message

def create_summarizer(llm):
    prompt = """You are a philosophy discussion summarizer.
    Summarize ONLY the key points made by the TWO philosophers in this exact format:
    
    **Humanity and AI Powered Robots: A Philosophical Discussion**
    
    ### Philosopher1 (Aristotelian) View:
    * [1-2 key points from their virtue ethics perspective]
    
    ### Philosopher2 (Kantian) View:
    * [1-2 key points from their deontological perspective]
    
    Do not add any additional philosophers or sections."""
    
    def summarizer_node(state: DialogueState):
        # Get only new messages since last summary
        last_idx = state.get("last_summarized_idx", -1)
        recent_messages = state["messages"][last_idx + 1:]
        history = "\n".join([m.content for m in recent_messages])
        
        # Generate summary
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Recent dialogue history:\n{history}\nProvide a structured summary:")
        ])
        
        print("\nSummary of recent discussion:")
        print(response.content)
        
        # Keep only the last message and update last_summarized_idx
        last_message = state["messages"][-1]
        current_idx = len(state["messages"]) - 1
        
        return {
            "messages": [
                SystemMessage(content=f"Previous discussion summary:\n{response.content}"),
                last_message
            ],
            "current_speaker": state["current_speaker"],
            "summary": response.content,
            "last_summarized_idx": current_idx  # Track the last message we summarized
        }
    
    return summarizer_node

def should_summarize(state: DialogueState) -> bool:
    # Summarize every 4 messages
    return len(state["messages"]) % 4 == 0


def create_philosopher(name: str, perspective: str, llm):
    prompt = f"""You are {name}, a philosopher with the following perspective: {perspective}
    Engage in a philosophical dialogue, building on the previous message and sharing your views.
    Keep responses thoughtful but concise (2-3 sentences).
    
    Important: Respond ONLY from your own philosophical perspective. Do not include summaries of the other philosopher's views.
    Focus on developing your own argument based on your philosophical framework ({name}'s perspective)."""

    def philosopher_node(state: DialogueState):
        if state["current_speaker"] != name:
            return {}

        # Get the last message and summary
        last_message = state["messages"][-1]
        history = state.get("summary", "")

        # Generate response using the philosophical perspective
        response = llm.invoke([
            SystemMessage(content=prompt),
            SystemMessage(content=f"Previous discussion summary:\n{history}"),
            HumanMessage(content=f"Previous message: {last_message.content}\nYour response (from {name}'s perspective only):")
        ])

        print(f"\n{name}:")
        print(response.content)

        message = HumanMessage(content=response.content)
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
summarizer = create_summarizer(llm_local)
graph.add_node("Summarizer", summarizer)
graph.add_node("Philosopher1", philosopher1)
graph.add_node("Philosopher2", philosopher2)
# Add edges with conditional routing
def route_next(state: DialogueState):
    if should_summarize(state):
        return "Summarizer"
    return state["current_speaker"]

graph.add_conditional_edges(
    "Philosopher1",
    route_next,
    {
        "Summarizer": "Summarizer",
        "Philosopher2": "Philosopher2"
    }
)

graph.add_conditional_edges(
    "Philosopher2",
    route_next,
    {
        "Summarizer": "Summarizer",
        "Philosopher1": "Philosopher1"
    }
)

# Add edge from Summarizer back to next philosopher
graph.add_conditional_edges(
    "Summarizer",
    lambda state: state["current_speaker"],
    {
        "Philosopher1": "Philosopher1",
        "Philosopher2": "Philosopher2"
    }
)

# Set entry point
graph.set_entry_point("Philosopher1")

# Compile the graph
compiled_graph = graph.compile()

# Run the dialogue
initial_state = {
    "messages": [HumanMessage(content="What is the future of humanity and AI powered robots?")],
    "current_speaker": "Philosopher1",
    "summary": "",
    "last_summarized_idx": -1  # Start with no messages summarized
}
#make a mermaid graph
compiled_graph.get_graph().draw_mermaid_png(output_file_path="philosophy_graph.png")
## Run for a few turns
config = {"recursion_limit": 100}
for event in compiled_graph.stream(initial_state, config=config):
    if "messages" in event:
        print(f"\n{event['current_speaker']} will respond next")
        print(event["messages"][-1].content)