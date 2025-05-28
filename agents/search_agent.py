import logging
import os
import json
from typing import Dict, Any, Optional, List

# Import for LiteLlm and Google's ADK
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import web search tool
from tools.web_search_tool import WebSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search-agent")

class SearchAgent:
    """
    Search Agent that performs web searches and formats the results
    
    This agent uses DuckDuckGo search to find information about any topic
    and formats the results in a clear, readable way.
    """
    
    def __init__(self):
        """Initialize the search agent with necessary tools"""
        # Initialize Groq model with LiteLlm
        self.groq_model = LiteLlm(
            # Use Llama 3 model from Groq
            model="groq/llama3-8b-8192",
            # Explicitly provide the API key from environment variables
            api_key=os.getenv("GROQ_API_KEY"),
            # Explicitly provide the Groq API base URL
            api_base="https://api.groq.com/openai/v1/chat/completions"
        )
        
        # Initialize the web search tool
        self.search_tool = WebSearchTool()
        
        # Session management
        self.session_service = InMemorySessionService()
        
        # Configure the LlmAgent for search results formatting
        self.search_agent = LlmAgent(
            model=self.groq_model,
            name="search_results_formatter",
            description="Formats search results in a clear, readable way.",
            instruction="""You are a helpful agent that formats search results.
            1. Analyze the search results provided.
            2. Provide a concise summary of the key information.
            3. Ensure accuracy and highlight the most relevant points.
            4. Format the response in a clear, readable way.
            """,
            tools=[self.search_tool.search],
            output_key="formatted_search_results",
        )
        
        # Create a runner for the agent
        self.runner = Runner(
            agent=self.search_agent,
            app_name="search_app",
            session_service=self.session_service
        )
        
        logger.info("SearchAgent initialized with Groq LLM and web search capabilities")
    
    async def process(self, exec_spec) -> Dict[str, Any]:
        """
        Process the execution request and return a response
        
        Args:
            exec_spec: The execution specification from the MCP server
            
        Returns:
            Dict containing the search results and formatted information
        """
        # Extract parameters from the execution request
        parameters = exec_spec.parameters or {}
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 8)
        detailed = parameters.get("detailed", False)
        
        logger.info(f"Processing search request for query: {query}")
        
        if not query:
            return {
                "success": False,
                "error": "No query provided",
                "message": "Please provide a query to search for"
            }
            
        try:
            # Step 1: Perform web search for the query
            search_results = await self._perform_search(query, num_results)
            
            # Step 2: Format the search results based on the detailed parameter
            formatted_results = await self._format_results(query, search_results, detailed)
            
            # Step 3: Return the combined results
            return {
                "success": True,
                "query": query,
                "search_results": search_results,
                "formatted_results": formatted_results,
                "search_engine": search_results.get("engine", "unknown"),
                "result_count": len(search_results.get("results", [])),
                "message": "Search completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing search request: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to perform search"
            }
    
    async def _perform_search(self, query: str, num_results: int = 8) -> Dict[str, Any]:
        """
        Perform the web search using DuckDuckGo
        
        Args:
            query: The search query string
            num_results: The number of results to return
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Use the async search if available
            if hasattr(self.search_tool, 'search_web'):
                search_results = await self.search_tool.search_web(query, num_results)
            else:
                # Fall back to synchronous search
                search_results = self.search_tool.search(query, num_results)
            
            logger.info(f"Found {len(search_results.get('results', []))} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return {
                "success": False,
                "message": f"Search error: {str(e)}",
                "results": []
            }
    
    async def _format_results(self, query: str, search_results: Dict[str, Any], detailed: bool) -> str:
        """
        Format the search results into a readable string using the Groq LLM agent
        
        Args:
            query: The original search query
            search_results: The results from the web search
            detailed: Whether to include detailed information
            
        Returns:
            Formatted string with the search results
        """
        # If the search failed, return error message
        if not search_results.get("success", False):
            return f"Search failed: {search_results.get('message', 'Unknown error')}"
        
        # Get the results
        results = search_results.get("results", [])
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format search results for the LLM
        context = "Search results:\n"
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            content = result.get("snippet", "") or result.get("description", "")
            link = result.get("link", "")
            context += f"{idx}. {title}: {content}\n"
            if link:
                context += f"   Source: {link}\n"
            context += "\n"
        
        # Create unique session ID for this request
        import uuid
        session_id = f"search_session_{str(uuid.uuid4())}"
        await self.session_service.create_session(app_name="search_app", user_id="search_user", session_id=session_id)
        
        try:
            # Prepare the user message with search results context
            from google.genai import types
            user_content = types.Content(
                role='user', 
                parts=[types.Part(text=f"""
                Query: {query}
                
                {context}
                
                Please format these search results in a clean, readable way. 
                {("Provide a detailed summary." if detailed else "Provide a brief summary.")}
                """)])
            
            # Run the agent
            final_response = ""
            async for event in self.runner.run_async(
                user_id="search_user", 
                session_id=session_id, 
                new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
            
            # If we got a response, use it; otherwise fall back to standard formatting
            if final_response:
                return final_response
                
        except Exception as e:
            logger.error(f"Error formatting results with LLM agent: {str(e)}")
            # Fall back to standard formatting
        
        # Standard formatting for results if LLM fails
        formatted_text = f"Search results for: {query}\n\n"
        
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "") or result.get("description", "")
            link = result.get("link", "")
            
            formatted_text += f"{idx}. {title}\n"
            if link:
                formatted_text += f"   URL: {link}\n"
            formatted_text += f"   {snippet}\n\n"
        
        return formatted_text