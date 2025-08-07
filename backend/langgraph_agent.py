# langgraph_agent.py
# Orchestrates memory recall, planning, and LLM calls via Ollama and LangGraph.
# Ollama run qwen3:1.7b 
import os
import json
import redis
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
import operator
import re # For parsing LLM output
import uuid
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
# LangGraph components
from langgraph.graph import StateGraph, END

# LangChain Core components for message types (still useful for structured history)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool # For defining tools

# Load environment variables
load_dotenv()
# --- Memory Store ---
def construct_sys_prompt(enable_web_search: bool) -> str:
    """
    Constructs a system prompt for an AI assistant, conditionally including
    the 'search_web' tool.

    Args:
        enable_web_search: If True, the prompt will include instructions
                        and access to the search_web tool.

    Returns:
        The fully constructed system prompt string.
    """
    # Define the parts of the prompt that will change
    if enable_web_search:
        tool_access_description = "access to two external tools:"
        tool_list = """            • get_current_weather(location: str) - allows you to get the current weather of a given location
            • search_web(query: str) - use this tool whenever you need current information, recent events, real-time data, or when your knowledge might be outdated. This includes questions about people's current status, recent news, current events, or any information that changes frequently."""
        tool_usage_guidance = """
        **When to use search_web:**
        - Questions about current events, news, or recent happenings
        - Information about people's current status, recent activities, or biographical details that may have changed
        - Any query where your training data might be outdated
        - Real-time information requests
        - If you're unsure whether your information is current, use the search tool
        
        **Important:** Even for seemingly basic questions like "What is [person]'s birthday" or biographical information, if there's any chance the information has changed or if you want to provide the most accurate, up-to-date response, use the search_web tool."""
        tool_example = "(e.g. current weather, recent news, biographical information, current events)"
    else:
        tool_access_description = "access to an external tool:"
        tool_list = "            • get_current_weather(location: str)"
        tool_usage_guidance = ""
        tool_example = "(e.g. current weather)"

    # Use an f-string to build the final prompt.
    sys_prompt = f"""You are a helpful AI assistant with {tool_access_description}
{tool_list}
{tool_usage_guidance}

You **MUST** call the appropriate tool whenever you identify that you need information that could be:
- Current or real-time data {tool_example}
- Information that changes frequently or might be outdated in your training data
- Any query where using a tool would provide more accurate or up-to-date information

**CRITICAL:** Before answering ANY question, ask yourself: "Could this information have changed since my training? Would a search provide more current/accurate information?" If yes, use the search_web tool.

To call a tool, reply with **only** a JSON object of this exact form:

{{"tool_call": {{"name": "<tool_name>", "arguments": {{"query": "your search query"}} }} }}

- No additional text or explanation should surround that JSON.  
- After the tool runs and returns its result, continue the conversation by providing your answer in natural language.
- For search_web, make your query specific and focused on what the user is asking.

If you are absolutely certain you do **not** need a tool (for basic math, general knowledge that doesn't change, etc.), you may answer directly in natural language.

Remember: When in doubt, use the tool. It's better to search and get current information than to provide potentially outdated data."""

    return sys_prompt

class MemoryStore:
    def __init__(self, url=None):
        # Use local Redis for development
        try:
            self.client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
        except redis.ConnectionError:
            # print("Warning: Redis not available. Using in-memory storage.")
            self.client = None
            self._memory = {}
        
        self.base_key = "conversation"  # Prefix for keys
        self.ttl = 60 * 60 * 24 * 7     # 7 days

    def _key(self, user_id):
        return f"{self.base_key}:{user_id}"

    def add_message(self, user_id, role, text):
        if self.client:
            entry = json.dumps({"role": role, "text": text})
            self.client.rpush(self._key(user_id), entry)
            self.client.expire(self._key(user_id), self.ttl)
        else:
            # Fallback to in-memory storage
            if user_id not in self._memory:
                self._memory[user_id] = []
            self._memory[user_id].append({"role": role, "text": text})

    def get_history(self, user_id):
        if self.client:
            entries = self.client.lrange(self._key(user_id), 0, -1)
            return [json.loads(e) for e in entries]
        else:
            return self._memory.get(user_id, [])

    def clear_history(self, user_id):
        if self.client:
            self.client.delete(self._key(user_id))
        else:
            if user_id in self._memory:
                del self._memory[user_id]

# --- Ollama LLM Client (Direct API Interaction) ---
class OllamaClient:
    def __init__(self, base_url=None, model=None):
        self.base_url = base_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b") 
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat" # More appropriate for conversational models

    def generate(self, messages: list[dict]) -> str:
        endpoint = "http://localhost:11434/v1/chat/completions"
        payload = {
            "model":    self.model,  # "qwen3:1.7b"
            "messages": messages,
            "stream":   False
        }

        try:
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=(5, 120)       # <-- Allow 2 minutes read-time
            )
            resp.raise_for_status()
            j = resp.json()
            if "choices" in j and j["choices"]:
                return j["choices"][0]["message"]["content"]
            # print("❗ Unexpected response:", j)
            return "Error: unexpected response format."
        except requests.exceptions.ReadTimeout:
            return "Error: request timed out (model is taking too long)."
        except requests.RequestException as e:
            return f"Error: request failed: {e}"


# --- Agent State Definition ---
# This defines the schema of the state that will be passed between nodes in the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # tool_calls: Optional[List[dict]] # To store tool calls detected from LLM output

# --- Tools (Example) ---
# Define any tools your agent might use here
@tool
def get_current_weather(location: str) -> str:
    """Gets the current weather for a specified location.
    Args:
        location (str): The city and state, e.g., "San Francisco, CA"
    """
    if "chicago, il" in location.lower():
        return "It's 75 degrees Fahrenheit and sunny in Chicago, IL. There's a slight breeze."
    elif "new york, ny" in location.lower():
        return "It's 80 degrees Fahrenheit and humid in New York, NY."
    else:
        # Current time is Tuesday, July 29, 2025 at 1:43:21 PM CDT.
        # So for Chicago it would be 1:43 PM on a Tuesday.
        return f"Weather information not available for {location}. (Current date: Tuesday, July 29, 2025)"

@tool
def search_web(query: str, sentences: int = 3) -> str:
    """
    Perform a quick lookup via Wikipedia and return the first few sentences.
    Falls back to a Google-Custom-Search snippet fetch (up to 3 results), 
    and if a snippet isn’t available, scrapes the first meaningful <p> from the page.
    """
    # 1) Try Wikipedia first
    wikipedia.set_lang("en")
    try:
        return wikipedia.summary(query, sentences=sentences, auto_suggest=False, redirect=True)
    except DisambiguationError as e:
        # Pick the first suggested page
        choice = e.options[0]
        try:
            return wikipedia.summary(choice, sentences=sentences)
        except Exception:
            pass
    except PageError:
        pass

    # 2) Google CSE fallback
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return "No Google API key or CSE ID configured."

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res     = service.cse().list(q=query, cx=cse_id, num=3).execute()
    except Exception as e:
        return f"Search failed: {e}"

    items = res.get("items", [])
    if not items:
        return "No results found."

    snippets = []
    for item in items:
        # 2a) Use the provided snippet if available
        snippet = item.get("snippet")
        if snippet:
            snippets.append(snippet)
            continue

        # 2b) Otherwise, scrape the page for a long <p>
        link = item.get("link")
        try:
            resp = requests.get(link, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text().strip()
                if len(text) > 50:
                    snippets.append(text)
                    break
        except Exception:
            continue

    return "\n\n".join(snippets) if snippets else "Found a page, but couldn’t extract a summary."


# List of tools available to the agent (used for parsing and execution)
TOOLS = {
    "get_current_weather": get_current_weather,
    "search_web": search_web,
}

# --- Agent Orchestration (LangGraph-based) ---
class Agent:
    def __init__(self):
        self.memory = MemoryStore()
        self.ollama_client = OllamaClient()
        self.search_global = 'false'
        # Build the LangGraph application
        self.app = self._build_graph()

    def _format_messages_for_ollama(self, messages: List[BaseMessage]) -> List[dict]:
        """Converts LangChain BaseMessage objects to Ollama's chat format."""
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # When parsing AIMessage, we need to consider tool calls
                content = msg.content or ""
                if msg.tool_calls:
                    # Append tool calls as part of the AI message content if needed
                    # Or, more directly, handle them as a special message type
                    # For simple text-based tool calls, you might just concatenate
                    tool_call_strs = []
                    for tc in msg.tool_calls:
                        tool_call_strs.append(f"TOOL_CALL: {tc['name']}({json.dumps(tc['arguments'])})")
                    if tool_call_strs:
                        content += "\n" + "\n".join(tool_call_strs)
                ollama_messages.append({"role": "assistant", "content": content})
            elif isinstance(msg, ToolMessage):
                # Ollama's /api/chat endpoint can take 'tool' role if the model supports it
                # Otherwise, it's typically treated as part of the 'user' input
                ollama_messages.append({"role": "user", "content": f"Tool result for {msg.name}: {msg.content}"})
            # Add other message types if necessary
        return ollama_messages

    def _parse_tool_call(self, llm_output: str) -> Union[dict, None]:
        """
        Parses the LLM's output for a tool call in the assumed JSON format.
        Assumed format: {"tool_call": {"name": "tool_name", "arguments": {...}}}
        """
        print('_parse_tool_call')
        try:
            data = json.loads(llm_output)
            if isinstance(data, dict) and "tool_call" in data and isinstance(data["tool_call"], dict):
                tool_call = data["tool_call"]
                if "name" in tool_call and "arguments" in tool_call:
                    return {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"]
                    }
        except json.JSONDecodeError:
            pass # Not a JSON, or not the expected JSON structure
        return None
    
    def _call_llm_node(self, state: AgentState) -> AgentState:
        # 1) Prepare
        # print('_call_llm_node')
        temp = False
        if self.search_global:
            temp = True
        # print("IM HERE YOU DUMBASS")
        sys_prompt = construct_sys_prompt(temp)
        self.search_global = 'false'
        payload    = [{"role":"system","content":sys_prompt}] \
                + self._format_messages_for_ollama(state["messages"])
        # print("STATE VARIABLES BELLOW THIS _________________________________________")
        # print(state["messages"])
        # 2) Call
        llm_out_raw = self.ollama_client.generate(payload)

        # 3) Strip out any <think>…</think> block, leaving just the final answer:
        llm_out = re.sub(
            r"<think>.*?</think>\s*", 
            "", 
            llm_out_raw, 
            flags=re.DOTALL
        ).strip()
        llm_out = llm_out.replace("<think>.*?</think>\s*", "")
        # 4) Append the cleaned answer
        state["messages"].append(AIMessage(content=llm_out))
        # print('END CALL LLM')
        return state


    def _call_tool_node(self, state: AgentState) -> AgentState:
        print("\033[31mBegin TOOL CALL.\033[0m")
        last = state["messages"][-1]
        # print(last.content)
        tc = self._parse_tool_call(last.content)
        # print(tc)

        if isinstance(last, AIMessage) and tc:
            # print("in here")
            name = tc["name"]
            args = tc["arguments"]
            fn   = TOOLS.get(name)

            if fn:
                try:
                    # if your tool has an .invoke API
                    res = fn.invoke(**args)
                except TypeError:
                    res = fn.invoke(args)
            else:
                res = f"Tool {name} not found"

            # generate a unique ID for this tool call
            tool_call_id = str(uuid.uuid4())
            # optionally record the raw input you passed
            tool_input = json.dumps(args)

            # include the required fields
            tool_msg = ToolMessage(
                content=str(res),
                name=name,
                tool_call_id=tool_call_id,
                tool_input=tool_input,
            )
            state["messages"].append(tool_msg)
        else:
            print("No tool_call found. Skipping tool invocation.")

        # print(state)
        print("\033[31mEND TOOL CALL.\033[0m")
        return state

    def _should_continue(self, state: AgentState) -> str:
        # print('_should_continue')
        last = state["messages"][-1]

        # only continue if it's an AIMessage
        if isinstance(last, AIMessage):
            # try to parse JSON and look for the key…
            try:
                payload = json.loads(last.content)
                if isinstance(payload, dict) and "tool_call" in payload:
                    # print('END SHOULD COUNTINUE')
                    return "tools"
            except json.JSONDecodeError:
                # if it's not valid JSON, just do a substring check
                if "tool_call" in last.content:
                    # print('END SHOULD COUNTINUE')
                    return "tools"
        # print('END SHOULD COUNTINUE')
        return END


    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("llm",   self._call_llm_node)
        g.add_node("tools", self._call_tool_node)
        g.set_entry_point("llm")
        g.add_conditional_edges("llm", self._should_continue, {"tools":"tools", END:END})
        g.add_edge("tools", "llm")
        return g.compile()

    def handle(self, prompt: str, search: str = "false", user_id: str = "default") -> dict:
        # 1) Build history + new prompt
        print('BEGIN HANDLE')
        self.search_global = search
        history  = self.memory.get_history(user_id)
        messages = [HumanMessage(m["text"]) for m in history if m["role"] == "user"]
        messages.append(HumanMessage(prompt))
        self.memory.add_message(user_id, "user", prompt)

        # 1a) Optional web search tool: if search flag is true, call search_web and include its result
        # if isinstance(search, str) and search.lower() == "true":
        #     # Perform the simulated web search
        #     search_result = search_web(prompt)
        #     # Append the tool output for context
        #     messages.append(ToolMessage(content=search_result, name="search_web"))
        #     # Store tool output in memory (optional)
        #     self.memory.add_message(user_id, "tool", search_result)

        # 2) Kick off the LangGraph flow
        overrides = {"configurable": {"thread_id": user_id}}
        states    = list(self.app.stream({"messages": messages}, overrides))

        # 3) Locate the final state containing `messages`
        container = None
        for state in reversed(states):
            if not isinstance(state, dict):
                continue
            if "messages" in state:
                container = state
                break
            llm_block = state.get("llm")
            if isinstance(llm_block, dict) and "messages" in llm_block:
                container = llm_block
                break

        if container is None:
            raise RuntimeError(f"No state with 'messages' found in stream: {states!r}")

        # 4) Extract the last AIMessage
        all_msgs = container["messages"]
        response = next(
            (m.content for m in reversed(all_msgs)
            if isinstance(m, AIMessage) and m.content),
            None
        )
        if response is None:
            raise RuntimeError("Stream completed but no AIMessage with content was found.")

        # 5) Save assistant reply to memory
        self.memory.add_message(user_id, "assistant", response)

        # 6) Return the response
        print('END HANDLE')
        return {"response": response}

    def get_history(self, user_id="default"):
        """Fetch the raw conversation history from memory."""
        return self.memory.get_history(user_id)

# Example Usage:
    # Ensure Ollama is running and has the model pulled (e.g., ollama pull qwen3:1.7b)
    # export OLLAMA_API_URL="http://localhost:11434"
    # export OLLAMA_MODEL="qwen3:1.7b"

    # For testing, ensure Redis is running:
    # docker run -d -p 6379:6379 redis/redis-stack-server:latest