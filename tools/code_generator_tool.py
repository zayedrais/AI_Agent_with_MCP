import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("code-generator-tool")

# --- Define Constants ---
APP_NAME = "mcp_code_generator_app"
USER_ID = "default_user"

class CodeGenerationInput(BaseModel):
    text: str = Field(description="The text prompt describing what code to generate.")

class CodeGenerationOutput(BaseModel):
    description: str = Field(description="A brief explanation of the code and how it works.")
    code: str = Field(description="The actual code snippet implementing the requested functionality.")
    language: str = Field(description="The programming language used in the code snippet.")

class CodeGeneratorTool:
    """
    A tool for generating code snippets using LLM models
    
    This tool provides functionality to generate code in various programming languages
    based on natural language descriptions.
    """
    
    def __init__(self, output_directory: str = None):
        """
        Initialize the code generator tool
        
        Args:
            output_directory: Directory to save generated code files. If None, use current directory/generated_code
        """
        logger.info("Initializing CodeGeneratorTool")
        
        # Main output directory is still generated_code
        self.output_directory = os.path.join(os.getcwd(), "generated_code")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            logger.info(f"Created AI code input directory at {self.output_directory}")
        
        # Initialize the LLM model using environment variables
        # First try Groq, fallback to OpenRouter if GROQ_API_KEY isn't available
        if os.getenv("OPENROUTER_API_KEY"):
            logger.info("Using OpenRouter LLM model")
            self.llm_model = LiteLlm(
                model="openrouter/deepseek/deepseek-chat-v3-0324:free",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                api_base="https://openrouter.ai/api/v1"
            )
        else:
            logger.warning("No API keys found for LLM services, code generation will fail")
            self.llm_model = None
            
        # Configure the LLM agent for code generation
        self.code_generation_agent = LlmAgent(
            model=self.llm_model,
            name="code_generation_agent",
            description="Generates code snippets with concise explanations",
            instruction="""You are a code generation agent that creates high-quality code examples.
The user will provide a text prompt describing what code they need.

Your response must include:
1. A brief explanation (3-5 sentences) of what the code does
2. The code snippet itself, properly formatted
3. The programming language used for the code

Prefer writing clean, maintainable, and well-commented code. Focus on best practices and include error handling where appropriate.

Your code should be clear, concise, and complete, implementing the full functionality described in the prompt.
If the language isn't specified, choose the most appropriate language for the task.

Return your response in the following format:
```json
{
  "description": "This function reads a CSV file and calculates statistics on numeric columns. It handles file not found errors and ignores non-numeric columns automatically.",
  "code": "import pandas as pd\\nimport numpy as np\\n\\ndef analyze_csv(filename):\\n    try:\\n        df = pd.read_csv(filename)\\n        numeric_cols = df.select_dtypes(include=[np.number]).columns\\n        \\n        if len(numeric_cols) == 0:\\n            return {'error': 'No numeric columns found'}\\n            \\n        results = {}\\n        for col in numeric_cols:\\n            results[col] = {\\n                'mean': df[col].mean(),\\n                'median': df[col].median(),\\n                'std': df[col].std()\\n            }\\n        return results\\n    except FileNotFoundError:\\n        return {'error': 'File not found: ' + filename}\\n    except Exception as e:\\n        return {'error': str(e)}",
  "language": "python"
}
```
""",
            input_schema=CodeGenerationInput,
            # Removed output_schema to use raw LLM response
            output_key="code_generation_result"
        )
        
        # Set up session management and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.code_generation_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )
        
        logger.info("CodeGeneratorTool initialized with LLM agent")
        
    async def generate_code(self, prompt: str, language: str = None) -> Dict[str, Any]:
        """
        Generate code based on a natural language prompt
        
        Args:
            prompt: Natural language description of the code to generate
            language: Optional specific programming language to use
            
        Returns:
            Dict containing the generated code and metadata
        """
        try:
            if not self.llm_model:
                return {
                    "success": False,
                    "error": "No LLM model available. Check API keys.",
                    "message": "Failed to generate code: LLM model not initialized"
                }
                
            logger.info(f"Generating code for prompt: {prompt[:50]}...")
            
            # Enhance the prompt with language specification if provided
            enhanced_prompt = prompt
            if language:
                enhanced_prompt = f"Generate code in {language}: {prompt}"
            
            # Create unique session ID for this request
            session_id = f"code_gen_session_{str(uuid.uuid4())}"
            self.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            
            # Create the user message
            input_json = {
                "text": enhanced_prompt
            }
            user_content = types.Content(role='user', parts=[types.Part(text=json.dumps(input_json))])
            
            # Process the request with the LLM agent
            final_response_content = None
            response_json = None
            
            async for event in self.runner.run_async(
                user_id=USER_ID, 
                session_id=session_id, 
                new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response_content = event.content.parts[0].text
            
            if final_response_content:
                # Try to parse the response as JSON
                try:
                    # First try to directly parse the response as JSON
                    response_json = json.loads(final_response_content)
                    logger.info("Successfully parsed direct JSON response")
                
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from the response
                    logger.warning("Failed to parse direct JSON response, trying alternative extraction methods")
                    
                    # Look for JSON object in the response using regex pattern matching
                    import re
                    
                    # Match JSON between triple backticks or outside of them
                    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})'
                    json_match = re.search(json_pattern, final_response_content, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                        try:
                            # Clean up the JSON string
                            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            
                            response_json = json.loads(json_str)
                            logger.info("Successfully parsed JSON from regex extraction")
                        except json.JSONDecodeError:
                            logger.error("Failed to parse extracted JSON")
                
                if not response_json:
                    # As a last resort, try to extract code blocks if JSON parsing failed
                    logger.warning("JSON parsing failed, attempting to extract code blocks directly")
                    
                    # Extract code blocks from the response using regex
                    code_pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
                    code_matches = re.findall(code_pattern, final_response_content)
                    
                    if code_matches:
                        # Identify the language if possible
                        lang_pattern = r'```(\w+)'
                        lang_match = re.search(lang_pattern, final_response_content)
                        detected_language = lang_match.group(1) if lang_match else "unknown"
                        
                        # Get the description (text before the first code block)
                        description_pattern = r'^(.*?)```'
                        description_match = re.search(description_pattern, final_response_content, re.DOTALL)
                        description = description_match.group(1).strip() if description_match else "Generated code"
                        
                        response_json = {
                            "description": description,
                            "code": code_matches[0]
                        }
                        logger.info("Extracted code blocks and created JSON structure")
                    else:
                        # If all parsing attempts fail, return the raw response
                        response_json = {
                            "description": "Generated code",
                            "code": final_response_content
                        }
                        logger.warning("Returning raw response as code")
            
            if response_json:
                # Extract code details from response
                code = response_json.get("code", "")
                description = response_json.get("description", "Generated code")
                detected_language = response_json.get("language", language) or "unknown"
                # Generate a file extension based on the language
                file_extension = self._get_extension_for_language(detected_language)
                
                # Create timestamp and clean name for this request
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_name = "".join(c if c.isalnum() else "_" for c in prompt[:20])
                
                # Create a unique folder for this request
                request_folder_name = f"{clean_name}_{timestamp}"
                request_folder_path = os.path.join(self.output_directory, request_folder_name)
                os.makedirs(request_folder_path, exist_ok=True)
                logger.info(f"Created request folder at {request_folder_path}")
                
                # Create a more descriptive code file name based on the prompt
                code_file_name = clean_name
                if len(code_file_name) > 30:
                    code_file_name = code_file_name[:30]
                
                # Save the code to a file in the request folder with the descriptive name and proper extension
                file_name = f"{code_file_name}.{file_extension}"
                file_path = os.path.join(request_folder_path, file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                logger.info(f"Generated code saved to {file_path}")
                
                # Save a metadata file for additional information
                metadata_file_name = f"{code_file_name}_metadata.json"
                metadata_file_path = os.path.join(request_folder_path, metadata_file_name)
                
                metadata = {
                    "timestamp": timestamp,
                    "prompt": prompt,
                    "generated_code": code,
                    "file_name": file_name
                }
                
                with open(metadata_file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Metadata saved to {metadata_file_path}")
                
                return {
                    "success": True,
                    "description": description,
                    "code": code,
                    "folder_path": request_folder_path,
                    "file_path": file_path,
                    "message": f"Successfully generated code in folder {request_folder_name}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to parse LLM response",
                    "message": "Could not extract code from the LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate code"
            }
    
    def _get_extension_for_language(self, language: str) -> str:
        """
        Get the appropriate file extension for a programming language
        
        Args:
            language: Programming language name
            
        Returns:
            File extension (without the dot)
        """
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "html": "html",
            "css": "css",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "csharp": "cs",
            "c#": "cs",
            "go": "go",
            "ruby": "rb",
            "php": "php",
            "swift": "swift",
            "kotlin": "kt",
            "rust": "rs",
            "scala": "scala",
            "sql": "sql",
            "shell": "sh",
            "bash": "sh",
            "powershell": "ps1",
            "r": "r",
            "json": "json",
            "yaml": "yaml",
            "markdown": "md"
        }
        
        return extensions.get(language.lower(), "txt")
    
    async def generate_code_with_requirements(
        self, 
        prompt: str, 
        language: str = None, 
        include_requirements: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code with package requirements (if applicable)
        
        Args:
            prompt: Natural language description of the code to generate
            language: Optional specific programming language to use
            include_requirements: Whether to include package requirements
            
        Returns:
            Dict containing the generated code, requirements, and metadata
        """
        try:
            # Generate the main code
            result = await self.generate_code(prompt, language)
            
            if not result.get("success", False):
                return result
            
            # Generate requirements if requested and applicable
            if include_requirements:
                detected_language = result.get("language", "unknown").lower()
                if detected_language in ["python", "javascript", "typescript", "ruby", "php"]:
                    requirements_prompt = f"Generate ONLY the package requirements for this {detected_language} code without explanations. Output just the requirements file content: {result.get('code', '')}"
                    
                    # Create requirements based on the language
                    if detected_language == "python":
                        result["requirements_file"] = "requirements.txt"
                        requirements_prompt = "Generate a requirements.txt file with package versions for this Python code: " + result.get('code', '')
                    elif detected_language in ["javascript", "typescript"]:
                        result["requirements_file"] = "package.json"
                        requirements_prompt = "Generate a minimal package.json file with dependencies for this JavaScript/TypeScript code: " + result.get('code', '')
                    elif detected_language == "ruby":
                        result["requirements_file"] = "Gemfile"
                        requirements_prompt = "Generate a Gemfile for this Ruby code: " + result.get('code', '')
                    elif detected_language == "php":
                        result["requirements_file"] = "composer.json"
                        requirements_prompt = "Generate a composer.json file for this PHP code: " + result.get('code', '')
                    
                    # Call LLM to generate requirements
                    requirements_result = await self.generate_code(requirements_prompt, detected_language)
                    
                    if requirements_result.get("success", False):
                        result["requirements"] = requirements_result.get("code", "")
                        
                        # Save requirements to file
                        req_file_path = os.path.join(self.output_directory, result["requirements_file"])
                        with open(req_file_path, 'w', encoding='utf-8') as f:
                            f.write(result["requirements"])
                        
                        result["requirements_path"] = req_file_path
                        logger.info(f"Generated requirements saved to {req_file_path}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating code with requirements: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate code with requirements"
            }