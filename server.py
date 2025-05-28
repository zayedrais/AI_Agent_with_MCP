from fastapi import FastAPI, Request
from fastmcp import FastMCP
import json
import asyncio
import tracemalloc
from typing import Dict, Any, List, Optional, Union
import logging
import datetime
import os

# Enable tracemalloc for memory leak debugging
tracemalloc.start()

# Import the coordinator agent process function
from agents.coordinator_agent import CoordinatorAgent

# Import the data analysis agent
from agents.data_analysis_agent import DataAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-server")

# File for logging requests and responses
LOG_FILE = "requests_log.txt"

# Create a FastAPI app
app = FastAPI(
    title="MCP Server for Tutorial Generation and Information Search",
    description="MCP Server using specialized agents for web search and tutorial creation",
    version="1.0.0",
)

# Create the FastMCP server
server = FastMCP(app=app)
# Note: transport-specific settings should be moved to run() call when starting the server

# Function to log interactions
async def log_interaction(description: str, response: Dict[str, Any]) -> None:
    """
    Logs the request and response to a file
    
    Args:
        description: Description of the request
        response: The response data
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"REQUEST: {description}\n\n")
        f.write(f"RESPONSE:\n{json.dumps(response, indent=2)}\n")
        f.write(f"{'='*80}\n")
    
    logger.info(f"Logged interaction to {LOG_FILE}")

# Initialize the coordinator agent for routing requests
coordinator = CoordinatorAgent()

# Define the function for handling routed requests
async def process_request(query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a request by routing it through the coordinator agent.
    
    Args:
        query: The user's query
        parameters: Additional parameters for processing
        
    Returns:
        Dict containing the response from the appropriate agent
    """
    logger.info(f"Processing request: {query}")
    
    if not parameters:
        parameters = {}
    
    # Ensure query is in parameters
    parameters["query"] = query
    
    try:
        # Create an execution spec with the parameters
        exec_spec = type('ExecSpec', (), {
            'name': 'coordinator_agent',
            'parameters': parameters
        })
        
        # Process the request with the coordinator agent and properly await the result
        result = await coordinator.process(exec_spec)
        
        if not result:
            response = {
                "success": False,
                "error": "Failed to process request. The query may be unclear or incomplete."
            }
            await log_interaction(query, response)
            return response
            
        # Log the interaction
        await log_interaction(query, result)
            
        # Return the result
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        response = {
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }
        await log_interaction(query, response)
        return response

# Define the MCP tool for general queries
@server.tool(
    name="ask",
    description="Routes the user's query to the most appropriate agent for processing"
)
async def ask_tool(query: str) -> Dict[str, Any]:
    """Tool handler for general queries"""
    return await process_request(query)

# Define the MCP tool for searches
@server.tool(
    name="search",
    description="Searches the web for information on any topic using search engines"
)
async def search_tool(query: str, detailed: bool = True) -> Dict[str, Any]:
    """Tool handler for search requests"""
    return await process_request(query, {"detailed": detailed})


# Define the MCP tool for data analysis
@server.tool(
    name="analyze_data",
    description="Analyzes data files and generates reports with visualizations"
)
async def analyze_data_tool(file_path: str = "") -> Dict[str, Any]:
    """Tool handler for data analysis requests"""
    return await process_request(f"Analyze data from {file_path}", {
        "file_path": file_path
    })

# Define the MCP tool for code generator
@server.tool(
    name="generate_code",
    description="Generates code in various programming languages based on natural language descriptions"
)
async def generate_code_tool(prompt: str, language: str = None, include_requirements: bool = True) -> Dict[str, Any]:
    """Tool handler for code generation requests"""
    return await process_request(prompt, {
        "prompt": prompt,
        "language": language,
        "include_requirements": include_requirements
    })

# Define custom route for MCP chat completion
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests directly"""
    body = await request.json()
    messages = body.get("messages", [])
    
    # Process the last user message
    last_user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
    
    if not last_user_message or not last_user_message.get("content"):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Please provide a query or question you'd like me to help with."
                    }
                }
            ]
        }
    
    user_query = last_user_message.get("content", "")
    
    # Check if this might be a data analysis request with a file_path
    parameters = {}
    file_path = body.get("file_path", "")
    if file_path and ("analyze" in user_query.lower() or "data" in user_query.lower()):
        logger.info(f"Chat completion: Data analysis request detected with file_path: {file_path}")
        # Update query to include file path if not already mentioned
        if file_path not in user_query:
            user_query = f"Analyze data from {file_path}"
        # Add file_path to parameters
        parameters["file_path"] = file_path
    
    # Call the process_request function
    try:
        # Use the coordinator to process the query
        result = await process_request(user_query, parameters)
        
        if "error" in result:
            # Return error message
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"I encountered an issue processing your request: {result['error']}"
                        }
                    }
                ]
            }
        
        # Format response
        agent_type = result.get("coordination", {}).get("routed_to", "unknown")
        
        # Prepare a human-readable response based on agent type
        if agent_type == "search_agent":
            search_results = result.get("results", [])
            if search_results:
                response = result.get("summary", "Here are the search results:")
                
                # Add reasoning from coordination if available
                reasoning = result.get("coordination", {}).get("reasoning", "")
                if reasoning:
                    response = f"{reasoning}\n\n{response}"
            else:
                response = "I couldn't find any relevant information for your query."
        
        elif agent_type == "web_tutorial_agent":
            tutorial = result.get("tutorial", {})
            tutorial_title = tutorial.get("title", "")
            
            if tutorial_title:
                response = f"I've created a tutorial on '{tutorial_title}'.\n\n"
                response += tutorial.get("content", "")
                
                # Add file path if saved to file
                file_path = result.get("file_path", "")
                if file_path:
                    response += f"\n\nThe tutorial has been saved to: {file_path}"
            else:
                response = "I couldn't generate a tutorial for your request."
        
        else:
            # Generic response for other agent types
            response = "Here's what I found for you:\n\n"
            response += json.dumps(result, indent=2)
        
        # Return the response with function call information
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response,
                        "tool_calls": [
                            {
                                "name": "process_request",
                                "arguments": {"query": user_query},
                                "response": result
                            }
                        ]
                    }
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in chat completions handler: {str(e)}")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error while processing your request: {str(e)}"
                    }
                }
            ]
        }
    


# Define the generic route for unified API access
@app.post("/ask")
async def ask_endpoint(request: Request):
    """
    Generic endpoint that routes requests to the appropriate specialized agent
    
    Args:
        request: The incoming request which should contain a 'query' field
        
    Returns:
        The response from the appropriate agent
    """
    try:
        # Parse the request body
        body = await request.json()
        query = body.get("query", "")
        
        if not query:
            return {
                "success": False,
                "message": "Please provide a 'query' parameter"
            }
        
        # Check for file_path parameter for data analysis
        file_path = body.get("file_path", "")
        if file_path and "analyze" in query.lower():
            logger.info(f"Data analysis request detected with file_path: {file_path}")
            # Update query to include file path if not already mentioned
            if file_path not in query:
                query = f"Analyze data from {file_path}"
            # Ensure file_path is in parameters
            body["file_path"] = file_path
        
        # Process the request
        result = await process_request(query, body)
        return result
        
    except Exception as e:
        logger.error(f"Error in ask_endpoint: {str(e)}")
        error_response = {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your request"
        }
        return error_response

# Define new endpoint for direct searching
@app.post("/search")
async def search_endpoint(request: Request):
    """
    Endpoint for direct searching
    
    Args:
        request: The incoming request which should contain a 'query' field
        
    Returns:
        Search results
    """
    try:
        # Parse the request body
        body = await request.json()
        query = body.get("query", "")
        detailed = body.get("detailed", True)
        
        if not query:
            return {
                "success": False,
                "message": "Please provide a 'query' parameter"
            }
        
        # Add parameters for search agent routing
        body["detailed"] = detailed
        
        # Process the request
        result = await process_request(query, body)
        return result
        
    except Exception as e:
        logger.error(f"Error in search_endpoint: {str(e)}")
        error_response = {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your search request"
        }
        return error_response

# Define custom route for data analysis
@app.post("/analyze-data")
async def analyze_data_endpoint(request: Request):
    """
    Handle data analysis requests
    
    Args:
        request: The incoming request
        
    Returns:
        The analysis results
    """
    try:
        body = await request.json()
        file_path = body.get("file_path", "")
        
        # Process the request - only file_path is needed now
        result = await process_request(f"Data analysis for {file_path}", {
            "file_path": file_path
        })
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_data_endpoint: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your data analysis request"
        }

# Define custom route for code generation
@app.post("/generate-code")
async def generate_code_endpoint(request: Request):
    """
    Handle code generation requests
    
    Args:
        request: The incoming request
        
    Returns:
        The generated code and metadata
    """
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        language = body.get("language", None)
        include_requirements = body.get("include_requirements", True)
        
        if not prompt:
            return {
                "success": False,
                "message": "Please provide a prompt describing what code you need"
            }
        
        # Add parameters for code generator agent routing
        parameters = {
            "prompt": prompt,
            "language": language,
            "include_requirements": include_requirements
        }
        
        # Process the request
        result = await process_request(prompt, parameters)
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_code_endpoint: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your code generation request"
        }

# Register all available agents and tools
def register_tools():
    """
    Register all available tools and agents for logging and tracking purposes
    
    Returns:
        List of registered tools and agents
    """
    items = [
        {
            "name": "web_search_tool",
            "description": "Search the web for information using search engines"
        },
        {
            "name": "data_reader_tool",
            "description": "Read data from various file formats (PDF, Excel, Word, CSV)"
        },
        {
            "name": "data_analysis_tool",
            "description": "Analyze data and generate visualizations"
        },
        {
            "name": "report_generator_tool",
            "description": "Generate formatted reports with data analysis results"
        },
        {
            "name": "code_generator_tool",
            "description": "Generates code in various programming languages using LLM models"
        },
        {
            "name": "coordinator_agent",
            "description": "LLM-based intelligent request router that directs queries to the appropriate specialized agent"
        },
        {
            "name": "search_agent",
            "description": "Performs web searches and provides information on any topic"
        },
        {
            "name": "data_analysis_agent",
            "description": "Analyzes data files and creates reports with visualizations"
        },
        {
            "name": "code_generator_agent",
            "description": "Generates code in various programming languages based on natural language descriptions"
        }
    ]
    
    # Log available tools and agents
    logger.info(f"Registered {len(items)} tools and agents")
    for item in items:
        logger.info(f"{item['name']} - {item['description']}")
    
    return items

if __name__ == "__main__":
    # Create log file if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write(f"Requests Log - Started {datetime.datetime.now()}\n")
    
    # Register tools and agents
    register_tools()
    
    # Define port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Start the server using uvicorn
    import uvicorn
    logger.info(f"Starting MCP Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)