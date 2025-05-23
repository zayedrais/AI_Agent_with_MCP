import logging
import os
from typing import Dict, Any, Optional
import asyncio

from tools.code_generator_tool import CodeGeneratorTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("code-generator-agent")

class CodeGeneratorAgent:
    """
    Agent that handles code generation requests using LLM models
    
    This agent utilizes CodeGeneratorTool to generate code in various programming languages
    based on natural language descriptions.
    """
    
    def __init__(self):
        """Initialize the code generator agent with required tools"""
        logger.info("Initializing CodeGeneratorAgent")
        self.code_generator_tool = CodeGeneratorTool()
    
    async def process(self, exec_spec) -> Dict[str, Any]:
        """
        Process the execution request by generating code from the prompt
        
        Args:
            exec_spec: The execution specification from the MCP server
            
        Returns:
            Dict containing the generated code or error information
        """
        # Extract parameters from the execution request
        parameters = exec_spec.parameters or {}
        prompt = parameters.get("prompt", "")
        language = parameters.get("language", None)
        include_requirements = parameters.get("include_requirements", True)
        
        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided",
                "message": "Please provide a prompt describing what code you need"
            }
        
        logger.info(f"Processing code generation request with prompt: {prompt[:50]}...")
        logger.debug(f"Parameters - language: {language}, include_requirements: {include_requirements}")
        
        try:
            # Generate code with requirements if requested
            if include_requirements:
                logger.info(f"Generating code with requirements...")
                result = await self.code_generator_tool.generate_code_with_requirements(
                    prompt=prompt,
                    language=language,
                    include_requirements=include_requirements
                )
            else:
                # Generate code without requirements
                logger.info(f"Generating code without requirements...")
                result = await self.code_generator_tool.generate_code(
                    prompt=prompt,
                    language=language
                )
                
            # If successful, format the response for better display
            if result.get("success", False):
                code = result.get("code", "")
                description = result.get("description", "")
                detected_language = result.get("language", "unknown")
                file_path = result.get("file_path", "")
                
                logger.info(f"Successfully generated {detected_language} code")
                
                # Format the response nicely
                response = {
                    "success": True,
                    "description": description,
                    "code": code,
                    "language": detected_language,
                    "file_path": file_path,
                    "message": f"Successfully generated {detected_language} code"
                }
                
                # Add requirements info if available
                if "requirements" in result:
                    response["requirements"] = result["requirements"]
                    response["requirements_file"] = result.get("requirements_file", "")
                    response["requirements_path"] = result.get("requirements_path", "")
            else:
                # Return error information
                response = {
                    "success": False,
                    "error": result.get("error", "Unknown error occurred"),
                    "message": result.get("message", "Failed to generate code")
                }
                
            return response
            
        except Exception as e:
            logger.error(f"Error in code generator agent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Code generation failed"
            }