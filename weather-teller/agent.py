# pip install groq python-dotenv requests
#
# .env file (same folder as this script):
#   GROQ_API_KEY=your_key_here
import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env into environment variables
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

# ---------- Constants & system prompt ----------
MODEL = "openai/gpt-oss-120b"  # any Groq model that supports tool calling
MAX_STEPS = 5

SYSTEM_PROMPT = """You are a helpful assistant.
- Never guess a live value. Call a tool for it.
- If a tool fails, say so. Do not invent a result.
"""

# ---------- Tool schema (OpenAI-style: wrapped in type/function) ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# ---------- Tool implementation ----------
def get_weather_tool(city: str) -> str:
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        if r.status_code == 200:
            return r.text
        return f"Failed to get weather for {city}. Status code: {r.status_code}"
    except requests.RequestException as e:
        return f"Failed to get weather for {city}. Error: {e}"

TOOLS = {"get_weather": get_weather_tool}

# ---------- The agent loop ----------
def run_task(question: str) -> str:
    client = Groq(api_key=API_KEY)

    # history = the agent's memory. System prompt is now the first message.
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")

        # PERCEIVE + REASON
        resp = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # No tool calls -> final answer
        if not msg.tool_calls:
            return msg.content

        # The model's proposal goes into history
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # ACT: your code executes each requested tool
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)  # arguments arrive as a JSON string
            print(f"Calling tool: {name} with args {args}")

            if name in TOOLS:
                result = TOOLS[name](**args)
            else:
                result = f"Error: unknown tool '{name}'"

            # OBSERVE: result goes back with role "tool", linked by tool_call_id
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return f"Stopped: hit MAX_STEPS of {MAX_STEPS}"

if __name__ == "__main__":
    answer = run_task("What is the weather in Murree,Lahore,Sheikhupura right now?")
    print("\nFINAL ANSWER:")
    print(answer)