# AI Agent Memory System

A conversational AI agent with persistent memory and web search capabilities built using LangGraph, Ollama, and Redis.

## Features

- **Persistent Memory**: Conversation history stored in Redis with automatic expiration
- **Web Search Integration**: Wikipedia and Google Custom Search API integration for real-time information
- **LangGraph Orchestration**: Structured conversation flow with tool calling capabilities
- **Ollama Integration**: Local LLM inference using Ollama
- **Weather Tool**: Get current weather information for any location
- **Modern Web Interface**: Clean, responsive frontend for easy interaction

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Infrastructure │
│   (HTML/CSS/JS) │◄──►│   (Flask)       │◄──►│   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   LangGraph     │
                       │   Agent         │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Ollama        │
                       │   (qwen3:1.7b)  │
                       └─────────────────┘
```

## Prerequisites

- Python 3.9+
- Redis server
- Ollama with qwen3:1.7b model
- Google API key (for web search functionality)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-agent-memory
   ```

2. **Set up Redis**
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis/redis-stack-server:latest
   
   # Or install locally
   brew install redis  # macOS
   sudo systemctl start redis  # Linux
   ```

3. **Install Ollama and pull the model**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the model
   ollama pull qwen3:1.7b
   ```

4. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   ```bash
   # Create .env file in backend directory
   cp .env.example .env
   
   # Edit .env with your settings
   OLLAMA_API_URL=http://localhost:11434
   OLLAMA_MODEL=qwen3:1.7b
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_custom_search_engine_id
   ```

## Usage

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```

2. **Open the frontend**
   - Navigate to `frontend/index.html` in your browser
   - Or serve it with a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```

3. **Start chatting!**
   - Type your message in the chat interface
   - Toggle "Enable Web Search" for real-time information
   - The agent will remember your conversation history

## API Endpoints

- `POST /chat` - Send a message to the agent
  - Parameters: `message` (string), `search` (boolean, optional)
  - Returns: JSON with `response` field

- `GET /history` - Get conversation history
  - Parameters: `user_id` (string, optional)
  - Returns: JSON array of conversation messages

## Tools Available

1. **get_current_weather(location: str)**
   - Gets current weather for a specified location
   - Example: "Chicago, IL"

2. **search_web(query: str)**
   - Performs web search using Wikipedia and Google Custom Search
   - Returns relevant information for current events and real-time data

## Configuration

### Environment Variables

- `OLLAMA_API_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: qwen3:1.7b)
- `GOOGLE_API_KEY`: Google API key for web search
- `GOOGLE_CSE_ID`: Google Custom Search Engine ID

### Redis Configuration

- Host: localhost
- Port: 6379
- TTL: 7 days (configurable in MemoryStore class)

## Development

### Project Structure

```
ai-agent-memory/
├── backend/
│   ├── app.py              # Flask application
│   ├── langgraph_agent.py  # Main agent implementation
│   ├── requirements.txt    # Python dependencies
│   └── venv/              # Virtual environment
├── frontend/
│   ├── index.html         # Main interface
│   ├── style.css          # Styling
│   └── script.js          # Frontend logic
├── infra/
│   └── redis-docker-compose.yml  # Redis setup
└── README.md
```

### Adding New Tools

1. Define the tool function with the `@tool` decorator
2. Add it to the `TOOLS` dictionary in `langgraph_agent.py`
3. Update the system prompt to include the new tool

### Customizing the Agent

- Modify `construct_sys_prompt()` to change agent behavior
- Adjust tool calling logic in `_parse_tool_call()`
- Customize memory storage in `MemoryStore` class

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running: `redis-cli ping`
   - Check if port 6379 is available

2. **Ollama Model Not Found**
   - Pull the model: `ollama pull qwen3:1.7b`
   - Check Ollama is running: `curl http://localhost:11434/api/tags`

3. **Web Search Not Working**
   - Verify Google API key and CSE ID are set
   - Check API quotas and billing

### Debug Mode

Enable debug logging by uncommenting print statements in `langgraph_agent.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph for conversation orchestration
- Ollama for local LLM inference
- Redis for persistent memory storage
- Wikipedia and Google APIs for web search capabilities
