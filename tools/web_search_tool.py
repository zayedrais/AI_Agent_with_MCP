import logging
import os
import json
#import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO)

class WebSearchTool:
    """
    A tool for performing web searches using DuckDuckGo
    
    This tool uses the duckduckgo_search library to search the web for information.
    No API key is required.
    """
    
    def __init__(self):
        """Initialize the web search tool with DuckDuckGo search"""
        # No API key needed for DuckDuckGo search
        pass
    
    async def search_web(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information on the given query using DuckDuckGo
        
        Args:
            query: The search query string
            num_results: The number of results to return (default: 5)
            
        Returns:
            Dict containing search results or error information
        """
        logging.info(f"Searching web using DuckDuckGo for: {query}")
        
        try:
            # Create a loop to run the synchronous DuckDuckGo search in a non-blocking way
            loop = asyncio.get_running_loop()
            ddg_results = await loop.run_in_executor(
                None, 
                lambda: DDGS().text(query, max_results=num_results)
            )
            
            # Format the results
            results = []
            for result in ddg_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "duckduckgo_search"
                })
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "engine": "duckduckgo"
            }
        
        except Exception as e:
            logging.error(f"Error performing DuckDuckGo web search: {str(e)}")
            return {
                "success": False,
                "message": f"Error performing web search: {str(e)}",
                "results": []
            }
    
    # For backward compatibility and synchronous usage
    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Synchronous wrapper around the async search_web method
        
        Args:
            query: The search query string
            num_results: The number of results to return (default: 5)
            
        Returns:
            Dict containing search results or error information
        """
        try:
            # Try to use DuckDuckGo search
            ddg_results = DDGS().text(query, max_results=num_results)
            
            # Format the results
            results = []
            for result in ddg_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "duckduckgo_search"
                })
            
            if results:
                return {
                    "success": True,
                    "results": results,
                    "count": len(results),
                    "engine": "duckduckgo"
                }
            else:
                # Fall back to mock search if no results
                logging.warning("No results from DuckDuckGo search, using mock search instead")
                mock_search = MockSearchTool()
                return mock_search.search(query)
                
        except Exception as e:
            logging.error(f"Error with DuckDuckGo search: {str(e)}, falling back to mock search")
            # Fall back to the mock knowledge base on error
            mock_search = MockSearchTool()
            return mock_search.search(query)


class MockSearchTool:
    """
    A simple mock search tool for use when the search API fails
    """
    
    def __init__(self):
        # Mock knowledge base about MCP concepts
        self.knowledge_base = {
            "mcp": "Model Context Protocol (MCP) is a protocol for contextual interactions between LLMs and tools/agents.",
            "agent": "In AI systems, an agent is a software entity that can perceive its environment, make decisions, and take actions.",
            "tool": "Tools are functions that agents can use to interact with external systems or perform specific tasks.",
            "context": "Context in MCP refers to the information available to the model during processing, which helps generate more relevant responses.",
            "fastmcp": "FastMCP is a Python library for building MCP-compatible servers easily.",
            "google adk": "Google's Agent Development Kit (ADK) provides tools and libraries for building AI agents using Google's models.",
            "web search": "Web search tools allow AI agents to retrieve current information from the internet to provide up-to-date responses.",
            "tutorial format": "A standard tutorial format includes an initial summary, prerequisites, introduction, setup and experimentation steps, conclusion, and references.",
            "duckduckgo": "DuckDuckGo is a privacy-focused search engine that can be used for web searches without requiring an API key."
        }
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information
        
        Args:
            query: The search query string
            
        Returns:
            Dict containing search results or error information
        """
        logging.info(f"Using mock search for: {query}")
        
        query = query.lower().strip()
        results = []
        
        # Simple keyword matching
        for key, value in self.knowledge_base.items():
            if query in key or any(term in value.lower() for term in query.split()):
                results.append({
                    "title": key,
                    "snippet": value,
                    "source": "mock_knowledge_base"
                })
        
        if results:
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "engine": "mock"
            }
        else:
            return {
                "success": False,
                "message": f"No results found for query: {query}",
                "results": []
            }