import logging
import os
from typing import Dict, Any, Optional
import json
import re

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coordinator-agent")

# --- Define Constants ---
APP_NAME = "mcp_coordinator_app"
USER_ID = "default_user"
SESSION_ID = "coordinator_session"

# --- Define Schemas ---
class CoordinatorInput(BaseModel):
    query: str = Field(description="The natural language query from the user")

class AgentRoutingOutput(BaseModel):
    agent_name: str = Field(
        description="The identified agent name based on the user query"
    )
    confidence_score: float = Field(
        description="Confidence score for the agent selection (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was selected"
    )

class CoordinatorAgent:
    """
    Coordinator Agent that analyzes requests and routes them to the appropriate agent
    using LLM-based intelligence rather than keyword matching.
    
    This agent uses Google's Agent Development Kit (ADK) LlmAgent to determine
    the best specialized agent to handle a request.
    """
    
    def __init__(self):
        """Initialize the coordinator agent with LLM capabilities"""
        # Initialize Groq model with LiteLlm
        self.groq_model = LiteLlm(
            # Use Llama 3 model from Groq
            model="groq/llama3-8b-8192",
            # Explicitly provide the API key from environment variables
            api_key=os.getenv("GROQ_API_KEY"),
            # Explicitly provide the Groq API base URL
            api_base="https://api.groq.com/openai/v1/chat/completions"
        )
        
        # Define available agents
        self.available_agents = [
            "search_agent",
            "data_analysis_agent",
            "code_generator_agent"
        ]
        
        # Configure the LLM agent for routing
        self.routing_agent = LlmAgent(
            model=self.groq_model,
            name="agent_routing",
            description="Routes user queries to the most appropriate specialized agent",
            instruction="""You are a coordinator agent that analyzes user queries and identifies which specialized agent should handle the request.

Currently, you can route to these agents:
1. search_agent - Performs web searches and provides information on any topic
2. data_analysis_agent - Analyzes data from files (PDF, Excel, Word, CSV) and generates reports with visualizations
3. code_generator_agent - Generates code in various programming languages based on natural language descriptions

SEARCH AGENT INDICATORS:
- Information-seeking queries about general topics
- Requests for facts, definitions, or current information
- Questions about "what is", "how does", "tell me about"
- Simple lookup requests with no need for code examples
- News, current events, or factual information needs

DATA ANALYSIS AGENT INDICATORS:
- Requests to analyze data, files, or documents
- Queries about generating reports from data
- Questions mentioning data formats like Excel, CSV, PDF, or Word
- Requests for data visualization, charts, or graphs
- Queries about extracting insights or summaries from datasets

CODE GENERATOR AGENT INDICATORS:
- Requests to generate, create, or write code
- Queries about programming solutions or implementations
- Mentions of specific programming languages (Python, JavaScript, etc.)
- Requests for code examples, functions, or algorithms
- Queries asking how to implement something with code

Analyze the user's query and determine which agent would best handle it. Return a JSON object with:
- agent_name: "search_agent", "data_analysis_agent", or "code_generator_agent"
- confidence_score: How confident you are (0.0-1.0)
- reasoning: Brief justification for your selection
""",
            input_schema=CoordinatorInput,
            output_schema=AgentRoutingOutput,
            output_key="agent_routing_decision",
        )
        
        # Set up session management and runner
        self.session_service = InMemorySessionService()
        # Create session asynchronously later in an async method instead of in __init__
        # Will be initialized in first process call
        
        # Create runner for the agent
        self.routing_runner = Runner(
            agent=self.routing_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )
        
        logger.info("CoordinatorAgent initialized with LLM routing capabilities")
    
    async def process(self, exec_spec) -> Dict[str, Any]:
        """
        Process the execution request by routing it to the appropriate agent
        
        Args:
            exec_spec: The execution specification from the MCP server
            
        Returns:
            Dict containing the response from the routed agent or error information
        """
        # Extract parameters from the execution request
        parameters = exec_spec.parameters or {}
        query = parameters.get("query", "")
        
        if not query:
            return {
                "success": False,
                "error": "No query provided",
                "message": "Please provide a query parameter to route your request"
            }
        
        logger.info(f"Processing coordinator request with query: {query}")
        
        try:
            # Step 1: Determine which agent should handle this request using LLM
            agent_name, confidence, reasoning = await self._route_with_llm(query)
            
            # Step 2: Format parameters for the selected agent
            agent_parameters = await self._format_parameters_for_agent(agent_name, query, parameters)
            
            # Step 3: Create a new execution spec for the selected agent
            agent_exec_spec = type('ExecSpec', (), {
                'name': agent_name,
                'parameters': agent_parameters
            })
            
            # Log the routing decision
            logger.info(f"Routing request to {agent_name} with confidence {confidence}")
            logger.info(f"Reasoning: {reasoning}")
            
            # Step 4: Import and invoke the selected agent
            agent_instance = await self._get_agent_instance(agent_name)
            if agent_instance:
                # Process the request with the selected agent
                response = await agent_instance.process(agent_exec_spec)
                
                # Add coordination metadata to the response
                if isinstance(response, dict):  # Ensure response is a dictionary
                    response["coordination"] = {
                        "routed_to": agent_name,
                        "confidence": confidence,
                        "reasoning": reasoning
                    }
                else:
                    # If response is not a dictionary, wrap it
                    response = {
                        "result": response,
                        "success": True,
                        "coordination": {
                            "routed_to": agent_name,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                    }
                
                return response
            else:
                return {
                    "success": False,
                    "error": f"Agent '{agent_name}' not found",
                    "message": "The selected agent could not be instantiated"
                }
        
        except Exception as e:
            logger.error(f"Error in coordinator agent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Coordination failed"
            }
    
    async def _route_with_llm(self, query: str) -> tuple:
        """
        Use LLM to determine which agent should handle the request
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (agent_name, confidence, reasoning)
        """
        logger.info(f"Routing query with LLM: {query}")
        
        # Create unique session ID for this request
        import uuid
        session_id = f"routing_session_{str(uuid.uuid4())}"
        # Use create_session_sync method to avoid asyncio issue
        await self.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        
        # Format the query as JSON input for the LLM agent with explicit instructions for proper JSON formatting
        enhanced_prompt = {
            "query": query,
            "instructions": """
Respond ONLY with a valid JSON object in the following format (no additional text before or after):
{
  "agent_name": "search_agent", // Choose one of: "search_agent", "data_analysis_agent", "code_generator_agent"
  "confidence_score": 0.95, // A float between 0.0 and 1.0
  "reasoning": "Brief explanation of why this agent was selected"
}
"""
        }
        query_json = json.dumps(enhanced_prompt)
        user_content = types.Content(role='user', parts=[types.Part(text=query_json)])
        
        # Default fallback values if LLM fails
        agent_name = "search_agent"
        confidence = 0.5
        reasoning = "Default fallback due to routing difficulties"
        
        try:
            final_response_content = None
            async for event in self.routing_runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response_content = event.content.parts[0].text
            
            if final_response_content:
                # First try to directly parse the response as JSON
                try:
                    result = json.loads(final_response_content)
                    if isinstance(result, dict) and "agent_name" in result:
                        agent_name = result.get("agent_name", "search_agent")
                        confidence = result.get("confidence_score", 0.5)
                        reasoning = result.get("reasoning", "Based on content analysis")
                        logger.info("Successfully parsed direct JSON response")
                    else:
                        raise ValueError("Response doesn't contain expected keys")
                
                except (json.JSONDecodeError, ValueError):
                    # If direct JSON parsing fails, try to extract JSON from various formats
                    logger.warning("Failed to parse direct JSON response, trying alternative extraction methods")
                    
                    # Look for JSON object in the response using regex pattern matching
                    import re
                    
                    # Match JSON between triple backticks or outside of them
                    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})'
                    json_match = re.search(json_pattern, final_response_content, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                        try:
                            # Clean up the JSON string - some models add trailing commas or comments
                            # Remove possible JavaScript comments
                            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                            # Remove possible trailing commas before closing brackets
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            
                            result = json.loads(json_str)
                            agent_name = result.get("agent_name", "search_agent")
                            confidence = result.get("confidence_score", 0.5)
                            reasoning = result.get("reasoning", "Based on content analysis")
                            logger.info("Successfully parsed JSON from regex extraction")
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse extracted JSON: {e}")
                            # Fall back to regex extraction of individual fields
                            logger.warning("Attempting to extract individual fields via regex")
                            
                    # If JSON parsing fails completely, use regex to extract individual fields
                    if agent_name == "search_agent" and confidence == 0.5 and reasoning == "Default fallback due to routing difficulties":
                        agent_match = re.search(r'"agent_name"\s*:\s*"([^"]+)"', final_response_content)
                        if agent_match and agent_match.group(1) in self.available_agents:
                            agent_name = agent_match.group(1)
                            logger.info(f"Extracted agent_name via field regex: {agent_name}")
                        
                        confidence_match = re.search(r'"confidence_score"\s*:\s*(0\.\d+|1\.0)', final_response_content)
                        if confidence_match:
                            try:
                                confidence = float(confidence_match.group(1))
                                logger.info(f"Extracted confidence_score via field regex: {confidence}")
                            except ValueError:
                                pass
                        
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', final_response_content)
                        if reasoning_match:
                            reasoning = reasoning_match.group(1)
                            logger.info(f"Extracted reasoning via field regex: {reasoning}")
                
                # Ensure the agent exists in our available agents
                if agent_name not in self.available_agents:
                    logger.warning(f"LLM suggested unknown agent: {agent_name}. Falling back to search_agent.")
                    agent_name = "search_agent"
                    confidence = 0.5
                    reasoning = f"Fallback from unknown agent: {agent_name}"
            else:
                logger.error("No final response from LLM router")
        
        except Exception as e:
            logger.error(f"Error in LLM routing: {str(e)}")
            # Continue with default fallback values
        
        logger.info(f"Final routing decision: {agent_name} (confidence: {confidence})")
        return agent_name, confidence, reasoning
    
    async def _format_parameters_for_agent(self, agent_name: str, query: str, original_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format parameters for the selected agent based on its requirements
        
        Args:
            agent_name: The name of the selected agent
            query: The user's query
            original_params: The original parameters provided in the request
            
        Returns:
            Dict of parameters formatted for the selected agent
        """
        # Start with a copy of the original parameters
        agent_params = original_params.copy()
        
        # Customize parameters based on the agent
        if agent_name == "search_agent":
            # Search agent needs a query
            if "query" not in agent_params:
                agent_params["query"] = query
            
            # Add detailed parameter if not present
            if "detailed" not in agent_params:
                agent_params["detailed"] = True
            
            # Ensure save_to_file parameter is set (default to True)
            if "save_to_file" not in agent_params:
                agent_params["save_to_file"] = True
        
        elif agent_name == "code_generator_agent":
            # Code generator agent needs a prompt
            if "prompt" not in agent_params:
                agent_params["prompt"] = query
                
            # Extract language if specified in the query
            language_match = re.search(r'in (\w+)(?:\s|:|$)', query, re.IGNORECASE)
            if language_match:
                potential_language = language_match.group(1).lower()
                common_languages = ["python", "javascript", "typescript", "java", "c#", "csharp", "go", "rust", "php", "ruby"]
                if potential_language in common_languages and "language" not in agent_params:
                    agent_params["language"] = potential_language
                    
            # Set requirement generation to be on by default
            if "include_requirements" not in agent_params:
                agent_params["include_requirements"] = True
        
        return agent_params
    
    async def _get_agent_instance(self, agent_name: str) -> Any:
        """
        Import and instantiate the requested agent
        
        Args:
            agent_name: The name of the agent to instantiate
            
        Returns:
            An instance of the requested agent or None if not found
        """
        try:
            if agent_name == "search_agent":
                from agents.search_agent import SearchAgent
                return SearchAgent()
                
            elif agent_name == "data_analysis_agent":
                from agents.data_analysis_agent import DataAnalysisAgent
                return DataAnalysisAgent()
            
            elif agent_name == "code_generator_agent":
                from agents.code_generator_agent import CodeGeneratorAgent
                return CodeGeneratorAgent()
            
            else:
                logger.error(f"Unknown agent: {agent_name}")
                return None
                
        except ImportError as e:
            logger.error(f"Error importing agent {agent_name}: {str(e)}")
            return None