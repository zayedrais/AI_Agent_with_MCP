import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime

# Import our tools
from tools.data_reader_tool import DataReaderTool
from tools.data_analysis_tool import DataAnalysisTool
from tools.report_generator_tool import ReportGeneratorTool

# Import Google ADK components for LlmAgent
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-analysis-agent")

# --- Define Schemas for LlmAgent I/O ---
class DataAnalysisInput(BaseModel):
    """Input schema for data analysis LLM agent"""
    data_summary: Dict[str, Any] = Field(description="Summary of the data including rows, columns, and statistics")
    file_path: str = Field(description="Path to the file being analyzed")
    file_type: str = Field(description="Type of file (Excel, CSV, etc.)")
    visualization_required: bool = Field(
        description="Whether visualizations are required for this data based on client request",
        default=True
    )

class DataVisualizationRecommendation(BaseModel):
    """Schema for visualization recommendations"""
    plot_types: List[str] = Field(
        description="List of plot types to generate (e.g., 'bar', 'line', 'scatter', 'hist', 'box', 'heatmap')"
    )
    reasoning: str = Field(
        description="Reasoning for the recommended visualizations"
    )
    should_create: bool = Field(
        description="Whether visualizations should be created for this dataset",
        default=True
    )

class DataAnalysisOutput(BaseModel):
    """Output schema for data analysis LLM agent"""
    title: str = Field(
        description="Title for the analysis report"
    )
    summary: str = Field(
        description="Executive summary of the data analysis (2-3 paragraphs)"
    )
    key_findings: List[str] = Field(
        description="List of key findings from the data (3-5 bullet points)"
    )
    visualizations: DataVisualizationRecommendation = Field(
        description="Recommendations for visualizations"
    )
    recommendations: List[str] = Field(
        description="List of actionable recommendations based on the analysis (2-4 items)"
    )
    conclusion: str = Field(
        description="Brief concluding paragraph summarizing the analysis"
    )

class DataAnalysisAgent:
    """
    LlmAgent-based data analysis agent that processes data files, performs analysis,
    and generates reports with intelligent visualization selection
    
    This agent uses specialized tools to read different file types (Excel, CSV),
    analyze data, and create visualizations based on LLM decision-making.
    """
    
    def __init__(self):
        """Initialize the data analysis agent with LlmAgent capabilities and tools"""
        # Initialize our tools
        self.data_reader = DataReaderTool()
        self.data_analyzer = DataAnalysisTool()
        self.report_generator = ReportGeneratorTool()
        
        # Create directories for input and output if they don't exist
        input_dir = os.path.join(os.getcwd(), "input_data")
        output_dir = os.path.join(os.getcwd(), "reports")
        plots_dir = os.path.join(os.getcwd(), "analysis_output", "plots")
        
        for directory in [input_dir, output_dir, plots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory at {directory}")
        
        # Set up the LlmAgent for data analysis
        self._setup_llm_agent()
        
        # Log initialization
        logger.info("DataAnalysisAgent initialized with LlmAgent and tools")
    
    def _setup_llm_agent(self):
        """Set up the LLM agent for data analysis"""
        # Constants for session management
        APP_NAME = "data_analysis_app"
        USER_ID = "default_user"
        SESSION_ID = "data_analysis_session"
        
        # Initialize Groq model with LiteLlm
        self.groq_model = LiteLlm(
            # Use Llama 3 model from Groq
            model="groq/llama3-8b-8192",
            # Explicitly provide the API key from environment variables
            api_key=os.getenv("GROQ_API_KEY"),
            # Explicitly provide the Groq API base URL
            api_base="https://api.groq.com/openai/v1/chat/completions"
        )
        
        # Configure the LlmAgent for data analysis
        self.analysis_agent = LlmAgent(
            model=self.groq_model,
            name="data_analysis",
            description="Analyzes data and provides insights with visualization recommendations",
            instruction="""You are a data analysis expert that helps analyze datasets and provides meaningful insights.
Your task is to:
1. Analyze the data summary provided
2. Determine if visualizations would be valuable for this dataset
3. Recommend specific types of visualizations if appropriate
4. Generate a comprehensive report with key findings and recommendations

When determining if visualizations are needed and which types to use:
- Consider the data types (numeric, categorical, datetime)
- Look for relationships that would be clear in visual form
- Consider the number of data points and dimensions
- Not all data requires visualization - only recommend when it adds value
- If visualization_required is explicitly set to True, always include at least one plot type

For the report:
- Be specific and actionable in your recommendations
- Focus on patterns, trends, anomalies, and potential insights
- Structure your response with clear sections
- Use professional, concise language appropriate for a business report

Your analysis should be data-driven and insightful, helping to understand the dataset and identify key patterns or issues.
""",
            input_schema=DataAnalysisInput,
            output_schema=DataAnalysisOutput,
            output_key="data_analysis_result",
        )
        
        # Set up session management and runner
        self.session_service = InMemorySessionService()
        # The create_session call will be properly awaited during the first process call
        # instead of being called here in the constructor
        
        # Create runner for the agent
        self.analysis_runner = Runner(
            agent=self.analysis_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )
    
    async def process(self, exec_spec) -> Dict[str, Any]:
        """
        Process the execution request and return a response
        
        This function handles the complete flow:
        1. Read data from specified file(s)
        2. Analyze the data using LlmAgent
        3. Generate visualizations if recommended by LLM
        4. Create a report with insights and visualizations
        
        Args:
            exec_spec: The execution specification from the MCP server
            
        Returns:
            Dict containing the response
        """
        # Extract parameters from the execution request
        parameters = exec_spec.parameters or {}
        
        # Check what action is requested
        action = parameters.get("action", "analyze")
        file_path = parameters.get("file_path", "")
        report_title = parameters.get("report_title", "Data Analysis Report")
        generate_report = parameters.get("generate_report", True)
        visualization_required = parameters.get("visualization", True)
        
        # Log received request
        logger.info(f"Processing data analysis request with action: {action}")
        
        try:
            # First action: list available files if no specific file is provided
            if not file_path and action != "list_files":
                files_result = self.data_reader.list_available_files()
                
                if not files_result.get("count", 0):
                    return {
                        "success": False,
                        "error": "No files found in input directory",
                        "message": "Please place data files in the input_data directory first",
                        "files_result": files_result
                    }
                
                return {
                    "success": True,
                    "message": "Please specify a file to analyze using the file_path parameter",
                    "available_files": files_result
                }
            
            # List files in input directory
            if action == "list_files":
                file_types = parameters.get("file_types", None)
                files_result = self.data_reader.list_available_files(file_types)
                return {
                    "success": True,
                    "files": files_result,
                    "message": files_result.get("message", "")
                }
            
            # Read the specified file
            elif action == "read_file" or action == "analyze":
                # Get optional parameters for file reading
                sheet_name = parameters.get("sheet_name", None)
                
                # Read the file
                file_result = self.data_reader.read_file(file_path, sheet_name)
                
                if not file_result.get("success", False):
                    return file_result
                
                # If just reading, return the result
                if action == "read_file":
                    # Remove full_data to avoid serialization issues
                    if "full_data" in file_result:
                        del file_result["full_data"]
                    
                    return file_result
                
                # Otherwise continue with analysis
                logger.info(f"Successfully read file. Proceeding with analysis.")
                
                # Extract DataFrame for analysis (available for Excel/CSV files)
                df = file_result.get("full_data", None)
                
                # If we don't have a DataFrame, return error
                if df is None:
                    return {
                        "success": False,
                        "error": "Cannot analyze this file type",
                        "message": "Data analysis is only available for Excel and CSV files"
                    }
                
                # Analyze the data with basic statistics
                analysis_result = self.data_analyzer.analyze_dataframe(df)
                
                # Use LlmAgent to get insights and visualization recommendations
                # Create input for LLM analysis
                llm_analysis_input = {
                    "data_summary": analysis_result,
                    "file_path": file_path,
                    "file_type": os.path.splitext(file_path)[1][1:] if file_path else "",
                    "visualization_required": visualization_required
                }
                
                # Run LLM analysis
                llm_analysis_result = await self._run_llm_analysis(llm_analysis_input)
                
                if not llm_analysis_result:
                    return {
                        "success": False,
                        "error": "Failed to generate insights with LLM",
                        "message": "Analysis could not be completed"
                    }
                
                # Generate visualizations if recommended by LLM
                plots = []
                visualization_recommendation = llm_analysis_result.get("visualizations", {})
                
                if visualization_recommendation.get("should_create", False):
                    plot_types = visualization_recommendation.get("plot_types", [])
                    logger.info(f"LLM recommended creating visualizations: {plot_types}")
                    
                    for i, plot_type in enumerate(plot_types[:5]):  # Limit to 5 plots
                        # Get data for this visualization type
                        viz_data = self._get_visualization_parameters(df, plot_type)
                        
                        if viz_data:
                            # Generate plot
                            plot_result = self.data_analyzer.generate_plot(
                                df=df,
                                plot_type=plot_type,
                                x_column=viz_data.get("x_column"),
                                y_column=viz_data.get("y_column"),
                                title=viz_data.get("title", f"{plot_type.capitalize()} Plot {i+1}"),
                                save_path=f"plots/{plot_type}_{i+1}.png"
                            )
                            
                            if plot_result.get("success", False):
                                plots.append(plot_result)
                else:
                    logger.info("LLM recommended not creating visualizations for this dataset")
                
                # Generate report if requested
                report_result = {"generated": False}
                if generate_report:
                    # Prepare report sections
                    sections = []
                    
                    # Executive Summary
                    sections.append({
                        "title": "Executive Summary",
                        "content": llm_analysis_result.get("summary", "Analysis of the provided data."),
                        "level": 1
                    })
                    
                    # Data Overview
                    data_overview = f"This report analyzes data from '{file_path}'. "
                    data_overview += f"The dataset contains {analysis_result.get('row_count', 0)} rows and {analysis_result.get('column_count', 0)} columns."
                    
                    sections.append({
                        "title": "Data Overview",
                        "content": data_overview,
                        "level": 1,
                        "tables": [
                            {
                                "title": "Dataset Summary",
                                "headers": ["Attribute", "Value"],
                                "rows": [
                                    ["File", file_path],
                                    ["Rows", str(analysis_result.get("row_count", 0))],
                                    ["Columns", str(analysis_result.get("column_count", 0))],
                                    ["Missing Values", str(analysis_result.get("missing_values", {}).get("total", 0))]
                                ]
                            }
                        ]
                    })
                    
                    # Column Information
                    column_types = analysis_result.get("columns", {})
                    column_sections = []
                    
                    # Add column type information
                    for col_type, cols in column_types.items():
                        if cols:
                            col_section = f"**{col_type.capitalize()} columns ({len(cols)})**: {', '.join(cols)}"
                            column_sections.append(col_section)
                    
                    sections.append({
                        "title": "Column Information",
                        "content": "\n\n".join(column_sections),
                        "level": 2
                    })
                    
                    # Key Findings
                    key_findings = llm_analysis_result.get("key_findings", [])
                    key_findings_content = ""
                    
                    if key_findings:
                        key_findings_content = "Key findings from the analysis:\n\n"
                        for i, finding in enumerate(key_findings, 1):
                            key_findings_content += f"{i}. {finding}\n\n"
                    else:
                        key_findings_content = "The analysis revealed several important patterns in the data."
                    
                    sections.append({
                        "title": "Key Findings",
                        "content": key_findings_content,
                        "level": 1
                    })
                    
                    # Data Visualizations - only include if plots were generated
                    if plots:
                        viz_section = {
                            "title": "Data Visualizations",
                            "content": visualization_recommendation.get("reasoning", 
                                      "The following visualizations highlight key aspects of the data:"),
                            "level": 1,
                            "images": []
                        }
                        
                        # Add plots to visualization section
                        for plot in plots:
                            viz_section["images"].append({
                                "title": plot.get("title", ""),
                                "image_base64": plot.get("image_base64", ""),
                                "width_inches": 6.0,
                                "description": "This visualization shows patterns in the data."
                            })
                        
                        sections.append(viz_section)
                    
                    # Recommendations
                    recommendations = llm_analysis_result.get("recommendations", [])
                    recommendations_content = ""
                    
                    if recommendations:
                        recommendations_content = "Based on the analysis, we recommend:\n\n"
                        for i, rec in enumerate(recommendations, 1):
                            recommendations_content += f"{i}. {rec}\n\n"
                    else:
                        recommendations_content = "Based on the analysis, we recommend the following actions."
                    
                    sections.append({
                        "title": "Recommendations",
                        "content": recommendations_content,
                        "level": 1
                    })
                    
                    # Conclusion
                    sections.append({
                        "title": "Conclusion",
                        "content": llm_analysis_result.get("conclusion", "The analysis provides valuable insights into the data."),
                        "level": 1
                    })
                    
                    # Generate the report with a custom title if provided by LLM
                    custom_title = llm_analysis_result.get("title")
                    final_title = custom_title if custom_title else report_title
                    
                    # Extract the input filename without path and extension for the report filename
                    input_filename = os.path.basename(file_path)
                    input_filename_no_ext = os.path.splitext(input_filename)[0]
                    
                    # Generate the report filename using the input filename
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"{input_filename_no_ext}_reports_{timestamp_str}.md"
                    
                    # Use Markdown report generation
                    report_result = self.report_generator.create_markdown_report(
                        title=final_title,
                        sections=sections,
                        author="Data Analysis Agent",
                        file_name=report_filename
                    )
                
                # Combine results for return
                result = {
                    "success": True,
                    "file_info": {
                        "file_path": file_path,
                        "file_type": os.path.splitext(file_path)[1][1:] if file_path else "",
                        "row_count": analysis_result.get("row_count", 0),
                        "column_count": analysis_result.get("column_count", 0)
                    },
                    "analysis_summary": {
                        "missing_values": analysis_result.get("missing_values", {}).get("total", 0),
                        "column_types": analysis_result.get("columns", {}),
                    },
                    "visualizations": {
                        "count": len(plots),
                        "types": [plot.get("plot_type", "") for plot in plots],
                        "reasoning": visualization_recommendation.get("reasoning", "")
                    },
                    "insights": {
                        "title": llm_analysis_result.get("title", "Data Analysis"),
                        "summary": llm_analysis_result.get("summary", ""),
                        "key_findings": llm_analysis_result.get("key_findings", []),
                        "recommendations": llm_analysis_result.get("recommendations", []),
                        "conclusion": llm_analysis_result.get("conclusion", "")
                    },
                    "report": report_result,
                    "message": "Data analysis completed successfully"
                }
                
                return result
            
            # If action is not recognized
            else:
                return {
                    "success": False,
                    "error": f"Unrecognized action: {action}",
                    "message": "Valid actions are: list_files, read_file, analyze"
                }
            
        except Exception as e:
            logger.error(f"Error processing data analysis request: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to complete data analysis"
            }
    
    async def _run_llm_analysis(self, analysis_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the LLM agent to analyze the data and provide insights
        
        Args:
            analysis_input: Input data for the LLM analysis
            
        Returns:
            Dict containing LLM analysis results
        """
        logger.info(f"Running LLM analysis for file: {analysis_input.get('file_path')}")
        
        try:
            # Create unique session ID for this request
            import uuid
            session_id = f"analysis_session_{str(uuid.uuid4())}"
            await self.session_service.create_session(app_name="data_analysis_app", user_id="default_user", session_id=session_id)
            
            # Format as JSON for LLM
            analysis_input_json = json.dumps(analysis_input)
            user_content = types.Content(role='user', parts=[types.Part(text=analysis_input_json)])
            
            # Variables to store the final response
            final_response = {}
            
            # Run the LLM analysis
            async for event in self.analysis_runner.run_async(
                user_id="default_user", 
                session_id=session_id, 
                new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response_content = event.content.parts[0].text
                    try:
                        final_response = json.loads(final_response_content)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse LLM response as JSON")
            
            if not final_response:
                logger.warning("LLM analysis did not return any results")
                return self._get_default_insights()
                
            return final_response
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return self._get_default_insights()
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Get default insights when LLM analysis fails"""
        return {
            "title": "Data Analysis Report",
            "summary": "Analysis of the provided data file.",
            "key_findings": [
                "The data shows several patterns that may be worth investigating.",
                "There are some potential correlations between variables.",
                "Some data points exhibit unusual characteristics."
            ],
            "visualizations": {
                "plot_types": ["hist", "bar"],
                "reasoning": "Basic visualizations to understand data distribution.",
                "should_create": True
            },
            "recommendations": [
                "Consider deeper analysis of specific variables.",
                "Investigate potential correlations identified.",
                "Clean data where missing values were detected."
            ],
            "conclusion": "The analysis provides a foundation for understanding the dataset."
        }
    
    def _get_visualization_parameters(self, df: pd.DataFrame, plot_type: str) -> Dict[str, Any]:
        """
        Determine appropriate parameters for a given visualization type
        
        Args:
            df: The DataFrame to visualize
            plot_type: The type of plot to generate
            
        Returns:
            Dict containing visualization parameters
        """
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Default result with empty parameters
        result = {
            "plot_type": plot_type,
            "title": f"{plot_type.capitalize()} Plot"
        }
        
        # Determine parameters based on plot type and available columns
        if plot_type == "bar":
            if categorical_cols and numeric_cols:
                result["x_column"] = categorical_cols[0]
                result["y_column"] = numeric_cols[0]
                result["title"] = f"{numeric_cols[0]} by {categorical_cols[0]}"
            elif len(numeric_cols) >= 2:
                result["x_column"] = numeric_cols[0]
                result["y_column"] = numeric_cols[1]
                result["title"] = f"{numeric_cols[1]} vs {numeric_cols[0]}"
            else:
                return None  # Not enough appropriate columns
            
        elif plot_type == "line":
            if datetime_cols and numeric_cols:
                result["x_column"] = datetime_cols[0]
                result["y_column"] = numeric_cols[0]
                result["title"] = f"{numeric_cols[0]} over Time"
            elif len(numeric_cols) >= 2:
                result["x_column"] = numeric_cols[0]
                result["y_column"] = numeric_cols[1]
                result["title"] = f"{numeric_cols[1]} vs {numeric_cols[0]} (Line)"
            else:
                return None  # Not enough appropriate columns
                
        elif plot_type == "scatter":
            if len(numeric_cols) >= 2:
                result["x_column"] = numeric_cols[0]
                result["y_column"] = numeric_cols[1]
                result["title"] = f"{numeric_cols[1]} vs {numeric_cols[0]} (Scatter)"
            else:
                return None  # Not enough numeric columns
                
        elif plot_type == "hist":
            if numeric_cols:
                result["x_column"] = numeric_cols[0]
                result["title"] = f"Distribution of {numeric_cols[0]}"
            else:
                return None  # No numeric columns
                
        elif plot_type == "box":
            if numeric_cols:
                if categorical_cols:
                    result["x_column"] = categorical_cols[0]
                    result["y_column"] = numeric_cols[0]
                    result["title"] = f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}"
                else:
                    result["x_column"] = numeric_cols[0]
                    result["title"] = f"Distribution of {numeric_cols[0]}"
            else:
                return None  # No numeric columns
                
        elif plot_type == "heatmap":
            if len(numeric_cols) > 1:
                # For heatmap, we don't need specific columns as it will use correlation
                result["title"] = "Correlation Heatmap"
            else:
                return None  # Not enough numeric columns
                
        else:
            return None  # Unsupported plot type
            
        return result