
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-12b",
    temperature=0
)

# Create Weather Agent
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""
    You are a helpful weather assistant.

    When the user asks about weather:
    - Identify the city.
    - Provide weather information if you know it.
    - Clearly state that you do not have live weather data.
    - Never invent current temperature or weather conditions.
    """
)

# Run the Agent
print("===== GROQ WEATHER AGENT =====")

while True:

    user_input = input("\nAsk about weather (or type exit): ")

    if user_input.lower() == "exit":
        print("Agent stopped.")
        break

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })

    print("\nAgent:", result["messages"][-1].content)
