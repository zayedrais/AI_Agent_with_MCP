import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-analysis-tool")

class DataAnalysisTool:
    """
    A tool for analyzing data and generating visualizations
    
    This tool provides functionality to analyze data from various sources,
    generate statistics, and create visualizations for reports.
    """
    
    def __init__(self, output_directory: str = None):
        """
        Initialize the data analysis tool
        
        Args:
            output_directory: Directory to save output files. If None, use current directory/analysis_output
        """
        logger.info("Initializing DataAnalysisTool")
        self.output_directory = output_directory or os.path.join(os.getcwd(), "analysis_output")
        logger.debug(f"Setting output directory to: {self.output_directory}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            logger.info(f"Created output directory at {self.output_directory}")
        
        # Set default plot style
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        logger.debug("Set default plotting style to whitegrid with figure size (10, 6)")
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a pandas DataFrame and generate summary statistics
        
        Args:
            df: The pandas DataFrame to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            logger.info(f"Analyzing DataFrame with shape {df.shape}")
            
            # Basic info
            row_count = len(df)
            col_count = len(df.columns)
            logger.debug(f"DataFrame has {row_count} rows and {col_count} columns")
            
            # Check for missing values
            missing_values = df.isnull().sum().to_dict()
            total_missing = sum(missing_values.values())
            logger.info(f"Found {total_missing} total missing values in the DataFrame")
            
            # Generate summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            logger.debug(f"Column types breakdown - numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, datetime: {len(datetime_cols)}")
            
            # Statistics for numeric columns
            numeric_stats = {}
            if numeric_cols:
                logger.info(f"Calculating statistics for {len(numeric_cols)} numeric columns")
                numeric_stats = df[numeric_cols].describe().to_dict()
            
            # Statistics for categorical columns
            categorical_stats = {}
            for col in categorical_cols:
                if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                    logger.debug(f"Analyzing categorical column: {col} with {df[col].nunique()} unique values")
                    value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 values
                    categorical_stats[col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": value_counts
                    }
            
            # Correlation matrix for numeric columns
            correlation_data = None
            if len(numeric_cols) > 1:
                logger.info(f"Calculating correlation matrix for {len(numeric_cols)} numeric columns")
                correlation_data = df[numeric_cols].corr().to_dict()
            
            logger.info("DataFrame analysis completed successfully")
            return {
                "success": True,
                "row_count": row_count,
                "column_count": col_count,
                "columns": {
                    "numeric": numeric_cols,
                    "categorical": categorical_cols,
                    "datetime": datetime_cols
                },
                "missing_values": {
                    "total": total_missing,
                    "by_column": missing_values
                },
                "numeric_statistics": numeric_stats,
                "categorical_statistics": categorical_stats,
                "correlation": correlation_data,
                "message": f"Successfully analyzed DataFrame with {row_count} rows and {col_count} columns"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing DataFrame: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to analyze DataFrame"
            }
    
    def generate_plot(self, 
                      df: pd.DataFrame, 
                      plot_type: str, 
                      x_column: str = None,
                      y_column: str = None,
                      title: str = None,
                      hue_column: str = None,
                      figsize: tuple = (10, 6),
                      save_path: str = None) -> Dict[str, Any]:
        """
        Generate a plot based on the provided DataFrame
        
        Args:
            df: The pandas DataFrame to plot
            plot_type: Type of plot ('bar', 'line', 'scatter', 'pie', 'hist', 'box', 'heatmap')
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis (optional for some plot types)
            title: Plot title
            hue_column: Column to use for color grouping (optional)
            figsize: Figure size as tuple (width, height) in inches
            save_path: Path to save the plot (optional)
            
        Returns:
            Dict containing the plot as base64 encoded image and metadata
        """
        try:
            logger.info(f"Generating {plot_type} plot")
            logger.debug(f"Plot parameters - x_column: {x_column}, y_column: {y_column}, title: {title}, hue_column: {hue_column}")
            
            # Create new figure
            plt.figure(figsize=figsize)
            logger.debug(f"Created figure with size {figsize}")
            
            # Generate plot based on type
            if plot_type == "bar":
                if x_column and y_column:
                    logger.debug(f"Creating bar plot with x={x_column}, y={y_column}")
                    sns.barplot(x=x_column, y=y_column, hue=hue_column, data=df)
                else:
                    logger.error("Bar plot creation failed: missing required parameters")
                    return {"success": False, "error": "Bar plot requires x_column and y_column"}
            
            elif plot_type == "line":
                if x_column and y_column:
                    logger.debug(f"Creating line plot with x={x_column}, y={y_column}")
                    sns.lineplot(x=x_column, y=y_column, hue=hue_column, data=df)
                else:
                    logger.error("Line plot creation failed: missing required parameters")
                    return {"success": False, "error": "Line plot requires x_column and y_column"}
            
            elif plot_type == "scatter":
                if x_column and y_column:
                    logger.debug(f"Creating scatter plot with x={x_column}, y={y_column}")
                    sns.scatterplot(x=x_column, y=y_column, hue=hue_column, data=df)
                else:
                    logger.error("Scatter plot creation failed: missing required parameters")
                    return {"success": False, "error": "Scatter plot requires x_column and y_column"}
            
            elif plot_type == "pie":
                if x_column and y_column:
                    # For pie chart, we need to pivot the data
                    logger.debug(f"Creating pie chart with labels={x_column}, values={y_column}")
                    plt.pie(df[y_column].values, labels=df[x_column].values, autopct='%1.1f%%')
                else:
                    logger.error("Pie chart creation failed: missing required parameters")
                    return {"success": False, "error": "Pie plot requires x_column (labels) and y_column (values)"}
            
            elif plot_type == "hist":
                if x_column:
                    logger.debug(f"Creating histogram for column {x_column}")
                    sns.histplot(x=x_column, hue=hue_column, data=df)
                else:
                    logger.error("Histogram creation failed: missing required parameters")
                    return {"success": False, "error": "Histogram requires x_column"}
            
            elif plot_type == "box":
                if x_column and y_column:
                    logger.debug(f"Creating box plot with x={x_column}, y={y_column}")
                    sns.boxplot(x=x_column, y=y_column, hue=hue_column, data=df)
                elif x_column:
                    logger.debug(f"Creating single-variable box plot for {x_column}")
                    sns.boxplot(x=x_column, data=df)
                else:
                    logger.error("Box plot creation failed: missing required parameters")
                    return {"success": False, "error": "Box plot requires at least x_column"}
            
            elif plot_type == "heatmap":
                if x_column and y_column:
                    # For heatmap, we need to pivot the data
                    logger.debug(f"Creating pivot table heatmap with index={x_column}, columns={y_column}")
                    pivot_table = df.pivot_table(index=x_column, columns=y_column, values=hue_column if hue_column else df.columns[0])
                    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
                else:
                    # Correlation heatmap
                    logger.debug("Creating correlation heatmap for numeric columns")
                    numeric_df = df.select_dtypes(include=[np.number])
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
            
            else:
                logger.error(f"Unsupported plot type: {plot_type}")
                return {"success": False, "error": f"Unsupported plot type: {plot_type}"}
            
            # Add title if provided
            if title:
                logger.debug(f"Setting plot title: {title}")
                plt.title(title)
            
            # Improve layout
            plt.tight_layout()
            logger.debug("Applied tight layout to the plot")
            
            # Save to buffer for base64 encoding
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.debug(f"Encoded plot to base64 string (length: {len(image_base64)})")
            
            # Save to file if path provided
            if save_path:
                full_path = os.path.join(self.output_directory, save_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                plt.savefig(full_path, dpi=100)
                logger.info(f"Saved plot to {full_path}")
            
            plt.close()
            logger.info(f"Successfully generated {plot_type} plot")
            
            return {
                "success": True,
                "plot_type": plot_type,
                "title": title,
                "image_base64": image_base64,
                "save_path": save_path if save_path else None,
                "message": f"Successfully generated {plot_type} plot"
            }
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}", exc_info=True)
            plt.close()
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate {plot_type} plot"
            }
    
    def recommend_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Recommend useful visualizations based on the data characteristics
        
        Args:
            df: The pandas DataFrame to analyze
            
        Returns:
            Dict containing recommended visualization options
        """
        try:
            logger.info(f"Recommending visualizations for DataFrame with {df.shape[1]} columns")
            
            recommendations = []
            
            # Get column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            logger.debug(f"Column type counts for visualization recommendations - numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, datetime: {len(datetime_cols)}")
            
            # Correlation heatmap for numeric columns
            if len(numeric_cols) > 1:
                logger.debug("Adding correlation heatmap to recommendations")
                recommendations.append({
                    "plot_type": "heatmap",
                    "title": "Correlation Matrix",
                    "description": "Visualize correlations between numeric variables"
                })
            
            # Distribution plots for numeric columns (limit to first 5)
            for col in numeric_cols[:5]:
                logger.debug(f"Adding histogram for numeric column: {col}")
                recommendations.append({
                    "plot_type": "hist",
                    "x_column": col,
                    "title": f"Distribution of {col}",
                    "description": f"Histogram showing the distribution of {col} values"
                })
            
            # Bar charts for categorical columns (limit to first 5)
            for col in categorical_cols[:5]:
                if df[col].nunique() < 20:  # Only for columns with reasonable number of categories
                    logger.debug(f"Adding bar chart for categorical column: {col} with {df[col].nunique()} categories")
                    recommendations.append({
                        "plot_type": "bar",
                        "x_column": col,
                        "y_column": df[col].value_counts().index.name or "count",
                        "title": f"Distribution of {col}",
                        "description": f"Bar chart showing the count of each {col} category"
                    })
            
            # Time series plots for datetime columns with numeric columns
            if datetime_cols and numeric_cols:
                for date_col in datetime_cols[:1]:  # Limit to first datetime column
                    for numeric_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        logger.debug(f"Adding time series line plot for {numeric_col} over {date_col}")
                        recommendations.append({
                            "plot_type": "line",
                            "x_column": date_col,
                            "y_column": numeric_col,
                            "title": f"{numeric_col} over Time",
                            "description": f"Line chart showing {numeric_col} values over time"
                        })
            
            # Scatter plots for pairs of numeric columns (limit to first 3 pairs)
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols[:3]):
                    for col2 in numeric_cols[i+1:i+3]:  # Take next 2 columns
                        logger.debug(f"Adding scatter plot for numeric columns: {col1} vs {col2}")
                        recommendations.append({
                            "plot_type": "scatter",
                            "x_column": col1,
                            "y_column": col2,
                            "title": f"{col1} vs {col2}",
                            "description": f"Scatter plot showing relationship between {col1} and {col2}"
                        })
            
            # Box plots for numeric columns across categories (if both present)
            if numeric_cols and categorical_cols:
                for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                    if df[cat_col].nunique() < 10:  # Only for columns with few categories
                        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                            logger.debug(f"Adding box plot for {num_col} grouped by {cat_col}")
                            recommendations.append({
                                "plot_type": "box",
                                "x_column": cat_col,
                                "y_column": num_col,
                                "title": f"{num_col} by {cat_col}",
                                "description": f"Box plot showing distribution of {num_col} for each {cat_col}"
                            })
            
            logger.info(f"Generated {len(recommendations)} visualization recommendations")
            return {
                "success": True,
                "recommendations": recommendations,
                "recommendation_count": len(recommendations),
                "message": f"Generated {len(recommendations)} visualization recommendations"
            }
            
        except Exception as e:
            logger.error(f"Error recommending visualizations: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to recommend visualizations"
            }
    
    def save_analysis_json(self, analysis_data: Dict[str, Any], file_name: str = "analysis_results.json") -> Dict[str, Any]:
        """
        Save analysis results to a JSON file
        
        Args:
            analysis_data: Analysis data to save
            file_name: Name of the output file
            
        Returns:
            Dict containing the result of the save operation
        """
        try:
            logger.info(f"Saving analysis results to JSON file: {file_name}")
            
            # Create a serializable copy of the data (remove DataFrame objects)
            serializable_data = {}
            for key, value in analysis_data.items():
                if key != "full_data":  # Exclude the DataFrame
                    serializable_data[key] = value
            
            logger.debug(f"Prepared serializable data with {len(serializable_data)} keys")
            
            # Save to file
            output_path = os.path.join(self.output_directory, file_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Saved analysis results to {output_path}")
            
            return {
                "success": True,
                "file_path": output_path,
                "message": f"Successfully saved analysis results to {output_path}"
            }
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save analysis results"
            }