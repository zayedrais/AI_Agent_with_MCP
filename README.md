# MCP Server with Google ADK Project

This project implements a Model Context Protocol (MCP) server using Google's Agent Development Kit (ADK) for building intelligent agents and tools. The system features multiple specialized agents that collaborate through a coordinator agent to handle different types of requests.

## Project Structure

```
├── agents/                    # Contains agent implementations
│   ├── coordinator_agent.py   # LLM-based intelligent request router
│   ├── data_analysis_agent.py # Agent for analyzing data files with visualization
│   ├── search_agent.py        # Agent for web searches and information retrieval
│   └── code_generator_agent.py # Agent for generating code based on descriptions
├── tools/                     # Contains tool implementations
│   ├── code_generator_tool.py # Generates code in various programming languages
│   ├── data_analysis_tool.py  # Analyzes data and creates visualizations
│   ├── data_reader_tool.py    # Reads data from various file formats
│   ├── report_generator_tool.py # Generates formatted reports
│   └── web_search_tool.py     # Performs web searches
├── input_data/                # Directory for input data files (CSV, Excel)
├── analysis_output/plots/     # Generated data visualizations
├── reports/                   # Generated analysis reports
├── generated_code/            # Generated code outputs
├── server.py                  # MCP Server implementation using FastMCP
├── requests_log.txt           # Log of requests and responses
└── README.md                  # This file
```

## Prerequisites

- Python 3.9+
- Groq and OpenRouter Key

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up your Groq & Openrouter Key:
  add key in .env if available otherwise create file as 
  ```
  touch .env 
  ```
  add key
   ```
   OPENROUTER_API_KEY="ADD_Your_Key"
   GROQ_API_KEY="ADD_Your_Key" 
   
   ```

## Running the MCP Server

```
python server.py
```

This will start the MCP server on `http://0.0.0.0:8080`

## Available Agents

### Coordinator Agent

The coordinator agent uses LLM-based routing to direct requests to the most appropriate specialized agent. It analyzes the content of the request and determines which agent can best handle it.

### Search Agent

Performs web searches and provides information on various topics.

**Example Request:**
```
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"query":"What is Model Context Protocol?"}'
```

### Data Analysis Agent

Analyzes data from various file formats (CSV, Excel) and generates reports with visualizations.

**Example Request:**
```
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"file_path": "sales_data.xlsx","query":"make a report for these data"}'
```

### Code Generator Agent

Generates code in various programming languages based on natural language descriptions.

**Example Request:**
```
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"query":"Python code for fibonacci series","language":"python"}'
```

## API Endpoints

- `/ask` - General endpoint that routes to the appropriate specialized agent
- `/search` - Endpoint for direct web searches
- `/analyze-data` - Endpoint for data analysis
- `/generate-code` - Endpoint for code generation
- `/chat/completions` - Chat completion endpoint for conversational interaction

## MCP Tools

The server exposes the following MCP tools:

- `ask` - Routes the user's query to the most appropriate agent
- `search` - Searches the web for information
- `analyze_data` - Analyzes data files and generates reports with visualizations
- `generate_code` - Generates code based on natural language descriptions

## Extending This Project

To add new agents:
1. Create a new file in the `agents/` directory
2. Implement the agent class with an async `process()` method
3. Update the coordinator agent to recognize and route to the new agent

To add new tools:
1. Create a new file in the `tools/` directory
2. Implement the tool functionality with comprehensive docstrings
3. Import and use the tool in your agents
