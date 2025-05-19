import pandas as pd

def load_and_clean_data(filepath, col_map):
    """
    Load and clean a CRM dataset with flexible column names.
    
    Parameters:
    - filepath: str
        Path to the Excel/CSV dataset.
    - col_map: dict
        Dictionary mapping logical field names to actual dataset column names.
        Required keys: 'customer_id', 'order_id', 'date', 'quantity', 'unit_price'
    
    Returns:
    - df: pd.DataFrame
        Cleaned DataFrame with standardized column names:
        ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'TotalPrice']
    """

    # Load dataset
    df = pd.read_excel(filepath)

    # Rename columns to standard internal names
    df = df.rename(columns={
        col_map['customer_id']: 'CustomerID',
        col_map['order_id']: 'InvoiceNo',
        col_map['date']: 'InvoiceDate',
        col_map['quantity']: 'Quantity',
        col_map['unit_price']: 'UnitPrice'
    })

    # Ensure required columns exist
    expected_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError("🚫 One or more required columns are missing after renaming.")

    # Drop missing CustomerID values
    df = df.dropna(subset=['CustomerID'])

    # Drop canceled transactions (if InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Convert date column to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Handle TotalPrice: calculate if not present
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    elif 'TotalPrice' in df.columns:
        pass  # already provided
    else:
        raise ValueError("Dataset must include either Quantity & UnitPrice or TotalPrice.")

    return df
