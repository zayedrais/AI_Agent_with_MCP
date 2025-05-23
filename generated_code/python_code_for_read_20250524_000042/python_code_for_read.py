import pandas as pd

def read_excel_to_dataframe(file_path):
    """
    Reads an Excel file and returns a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file (.xls or .xlsx)
    
    Returns:
        pd.DataFrame or dict: DataFrame if successful, error dict if failed
    """
    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        return {'error': f'File not found: {file_path}'}
    except Exception as e:
        return {'error': f'Failed to read Excel file: {str(e)}'}

# Example usage:
# df = read_excel_to_dataframe('data.xls')
# if isinstance(df, pd.DataFrame):
#     print(df.head())