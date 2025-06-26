#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:,.0f}' if abs(x) > 1 else f'{x:.4f}')


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    base = Path(filename).stem
    parts = base.split('_')

    if len(parts) == 4:
        return {
            'company': parts[0],
            'year': int(parts[1]),
            'quarter': parts[2],
            'filing_type': parts[3],
            'period': f"{parts[1]}-{parts[2]}",
            'filename': filename
        }
    elif len(parts) == 3:
        return {
            'company': parts[0],
            'year': int(parts[1]),
            'quarter': None,
            'filing_type': parts[2],
            'period': parts[1],
            'filename': filename
        }
    return None

# Load an Excel file and return a DataFrame for a specific sheet
def load_financial_statement(file_path: str, primary_sheet: str,
                           alt_sheet: Optional[str] = None) -> Optional[pd.DataFrame]:
    for sheet_name in [primary_sheet, alt_sheet]:
        if sheet_name:
            try:
                return pd.read_excel(file_path, sheet_name=sheet_name)
            except:
                continue
    return None


def process_financial_dataframe(df: pd.DataFrame, file_info: Dict) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    df_processed = df.copy()
    for key, value in file_info.items():
        if key != 'filename':
            df_processed[key] = value
    df_processed['source_file'] = file_info['filename']

    return df_processed


def create_fiscal_date(year: int, quarter: Optional[str]) -> str:
    """
    Convert year and quarter to YYYY-MM-DD format using Microsoft fiscal calendar.

    Microsoft's fiscal year runs July-June:
    - FY2014 Q1 ends September 30, 2013
    - FY2014 Q2 ends December 31, 2013
    - FY2014 Q3 ends March 31, 2014
    - FY2014 Q4 ends June 30, 2014
    """
    quarter_dates = {
        'Q1': f"{year-1}-09-30",
        'Q2': f"{year-1}-12-31",
        'Q3': f"{year}-03-31",
        'Q4': f"{year}-06-30",
        None: f"{year}-06-30"  # Annual reports contain information that help to calculate Q4
    }
    return quarter_dates.get(quarter)

# Extract specific income statement metrics
def extract_income_metrics(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()
    df_clean.rename(columns={df_clean.columns[0]: 'Metric'}, inplace=True)

    df_clean['Metric'] = df_clean['Metric'].astype(str).str.strip()
    df_clean = df_clean[
        df_clean['Metric'].notna() &
        (df_clean['Metric'] != 'nan') &
        (df_clean['Metric'] != '')
    ].reset_index(drop=True)

    extracted_rows = []
    processed_indices = set()

    # Revenue extraction, complicated due to the change in naming convention in the Excel files"
    revenue_metrics = ['total revenue', 'total revenues', 'net revenue', 'net revenues', 'revenue', 'revenues']
    for metric_name in revenue_metrics:
        for i, row in df_clean.iterrows():
            if i in processed_indices:
                continue
            if row['Metric'].lower() == metric_name:
                if metric_name in ['revenue', 'revenues']:
                    # Check if there's a value
                    if len(row) <= 1 or pd.isna(row.iloc[1]):
                        continue
                row_copy = row.copy()
                row_copy['Metric'] = 'Revenue'
                extracted_rows.append(row_copy)
                processed_indices.add(i)
                break
        if processed_indices:
            break

    # Cost of revenue - as above
    cost_metrics = ['total cost of revenue', 'cost of revenue', 'cost of goods sold']
    for metric_name in cost_metrics:
        for i, row in df_clean.iterrows():
            if i in processed_indices:
                continue
            if metric_name in row['Metric'].lower():
                if metric_name != 'total cost of revenue' and (len(row) <= 1 or pd.isna(row.iloc[1])):
                    continue
                row_copy = row.copy()
                row_copy['Metric'] = 'Cost of Revenue'
                extracted_rows.append(row_copy)
                processed_indices.add(i)
                break
        if any('Cost of Revenue' in r['Metric'] for r in extracted_rows):
            break

    #  Metric mappings
    metric_mappings = {
        'gross margin': 'Gross Margin',
        'gross profit': 'Gross Margin',
        'research and development': 'Research and Development',
        'sales and marketing': 'Sales and Marketing',
        'general and administrative': 'General and Administrative',
        'operating income': 'Operating Income',
        'other income': 'Other Income (Expense), Net',
        'other expense': 'Other Income (Expense), Net',
        'income before income taxes': 'Income Before Income Taxes',
        'provision for income taxes': 'Provision for Income Taxes',
        'net income': 'Net Income',
        'net income (loss)': 'Net Income'
    }

    # Extract the metrics here
    for i, row in df_clean.iterrows():
        if i in processed_indices:
            continue

        metric_lower = row['Metric'].lower()

        for pattern, target_metric in metric_mappings.items():
            if pattern in metric_lower:
                # Check if the metric already exists
                if not any(r['Metric'] == target_metric for r in extracted_rows):
                    row_copy = row.copy()
                    row_copy['Metric'] = target_metric
                    extracted_rows.append(row_copy)
                    processed_indices.add(i)
                    break

    # Handle EPS and Share counts
    for i, row in df_clean.iterrows():
        if i in processed_indices:
            continue

        metric_lower = row['Metric'].lower()

        # Look for EPS section
        if ('earnings' in metric_lower and 'per share' in metric_lower) or 'loss) per share' in metric_lower:
            if i + 2 < len(df_clean):
                # Next row should be Basic
                if df_clean.iloc[i+1]['Metric'].lower() == 'basic' and (i+1) not in processed_indices:
                    row_copy = df_clean.iloc[i+1].copy()
                    row_copy['Metric'] = 'Basic Earnings Per Share'
                    extracted_rows.append(row_copy)
                    processed_indices.add(i+1)

                # Row after that should be Diluted
                if df_clean.iloc[i+2]['Metric'].lower() == 'diluted' and (i+2) not in processed_indices:
                    row_copy = df_clean.iloc[i+2].copy()
                    row_copy['Metric'] = 'Diluted Earnings Per Share'
                    extracted_rows.append(row_copy)
                    processed_indices.add(i+2)

        # Look for Shares Outstanding section
        elif 'shares outstanding' in metric_lower:
            if i + 2 < len(df_clean):
                # Next row should be Basic
                if df_clean.iloc[i+1]['Metric'].lower() == 'basic' and (i+1) not in processed_indices:
                    row_copy = df_clean.iloc[i+1].copy()
                    row_copy['Metric'] = 'Weighted-Average Shares Outstanding - Basic'
                    extracted_rows.append(row_copy)
                    processed_indices.add(i+1)

                # Row after that should be Diluted
                if df_clean.iloc[i+2]['Metric'].lower() == 'diluted' and (i+2) not in processed_indices:
                    row_copy = df_clean.iloc[i+2].copy()
                    row_copy['Metric'] = 'Weighted-Average Shares Outstanding - Diluted'
                    extracted_rows.append(row_copy)
                    processed_indices.add(i+2)

    if extracted_rows:
        result_df = pd.DataFrame(extracted_rows)
        # Add metadata
        metadata_cols = ['source_file', 'year', 'quarter', 'period', 'filing_type']
        for col in metadata_cols:
            if col in df.columns:
                result_df[col] = df[col].iloc[0]
        return result_df
    return pd.DataFrame()

# Extract balance sheet metrics
def extract_balance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.rename(columns={df_clean.columns[0]: 'Metric'}, inplace=True)
    df_clean['Metric'] = df_clean['Metric'].astype(str).str.strip()
    df_clean = df_clean[
        df_clean['Metric'].notna() &
        (df_clean['Metric'] != 'nan') &
        (df_clean['Metric'] != '')
    ]

    metric_searches = [
        ('Cash and Cash Equivalents', lambda x: 'cash and cash equivalents' in x.lower()),
        ('Short-Term Investments', lambda x: 'short-term investments' in x.lower() or 'marketable securities' in x.lower()),
        ('Accounts Receivable, Net', lambda x: 'accounts receivable' in x.lower() and 'net' in x.lower()),
        ('Inventories', lambda x: x.lower() in ['inventories', 'inventory']),
        ('Total Current Assets', lambda x: 'total current assets' in x.lower()),
        ('Property and Equipment, Net', lambda x: 'property and equipment' in x.lower() and 'net' in x.lower()),
        ('Goodwill', lambda x: x.lower() == 'goodwill'),
        ('Intangible Assets, Net', lambda x: 'intangible assets' in x.lower() and 'net' in x.lower()),
        ('Total Assets', lambda x: x.lower() == 'total assets'),
        ('Accounts Payable', lambda x: x.lower() == 'accounts payable'),
        ('Accrued Compensation', lambda x: 'accrued compensation' in x.lower()),
        ('Unearned Revenue', lambda x: 'unearned revenue' in x.lower() or 'deferred revenue' in x.lower()),
        ('Total Current Liabilities', lambda x: 'total current liabilities' in x.lower()),
        ('Long-Term Debt', lambda x: 'long-term debt' in x.lower()),
        ('Total Liabilities', lambda x: x.lower() == 'total liabilities'),
        ('Common Stock', lambda x: 'common stock' in x.lower()),
        ('Retained Earnings', lambda x: 'retained earnings' in x.lower() or 'retained deficit' in x.lower()),
        ('Total Stockholders Equity', lambda x: all(word in x.lower().replace("'", "")
                                                   for word in ['stockholders', 'equity', 'total'])),
    ]

    extracted_rows = []
    for target_metric, search_func in metric_searches:
        matching_rows = df_clean[df_clean['Metric'].apply(search_func)]
        if not matching_rows.empty:
            row = matching_rows.iloc[0].copy()
            row['Metric'] = target_metric
            extracted_rows.append(row)

    if extracted_rows:
        result_df = pd.DataFrame(extracted_rows)
        metadata_cols = ['source_file', 'year', 'quarter', 'period', 'filing_type']
        for col in metadata_cols:
            if col in df.columns:
                result_df[col] = df[col].iloc[0]
        return result_df
    return pd.DataFrame()

# Extract cash flow metrics
def extract_cashflow_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.rename(columns={df_clean.columns[0]: 'Metric'}, inplace=True)
    df_clean['Metric'] = df_clean['Metric'].astype(str).str.strip()
    df_clean = df_clean[
        df_clean['Metric'].notna() &
        (df_clean['Metric'] != 'nan') &
        (df_clean['Metric'] != '')
    ]

    metric_searches = [
        ('Net Income', lambda x: x.lower() == 'net income'),
        ('Depreciation, Amortization, and Other', lambda x: 'depreciation' in x.lower() and 'amortization' in x.lower()),
        ('Net Cash from Operations', lambda x: x.lower().strip() == 'net cash from operations'),
        ('Change in Accounts Receivable', lambda x: x.lower().strip() == 'accounts receivable'),
        ('Change in Inventories', lambda x: x.lower().strip() == 'inventories'),
        ('Change in Accounts Payable', lambda x: x.lower().strip() == 'accounts payable'),
        ('Purchases of Property and Equipment', lambda x: 'additions to property' in x.lower() or 'purchases of property' in x.lower()),
        ('Net Cash Used in Investing', lambda x: 'net cash' in x.lower() and 'investing' in x.lower()),
        ('Net Cash Used in Financing', lambda x: 'net cash' in x.lower() and 'financing' in x.lower()),
        ('Dividends Paid', lambda x: 'dividends' in x.lower() and ('paid' in x.lower() or 'common stock' in x.lower())),
        ('Repurchases of Common Stock', lambda x: 'repurchase' in x.lower() and 'common stock' in x.lower()),
        ('Effect of Foreign Exchange Rates', lambda x: 'effect' in x.lower() and 'exchange' in x.lower() and 'cash' in x.lower()),
        ('Net Change in Cash and Cash Equivalents', lambda x: 'net change in cash' in x.lower() or 'net increase' in x.lower()),
        ('Cash and Cash Equivalents - Beginning', lambda x: 'cash' in x.lower() and 'equivalent' in x.lower() and 'beginning' in x.lower()),
        ('Cash and Cash Equivalents - End', lambda x: 'cash' in x.lower() and 'equivalent' in x.lower() and 'end' in x.lower()),
    ]

    extracted_rows = []
    for target_metric, search_func in metric_searches:
        matching_rows = df_clean[df_clean['Metric'].apply(search_func)]
        if not matching_rows.empty:
            row = matching_rows.iloc[0].copy()
            row['Metric'] = target_metric
            extracted_rows.append(row)

    if extracted_rows:
        result_df = pd.DataFrame(extracted_rows)
        metadata_cols = ['source_file', 'year', 'quarter', 'period', 'filing_type']
        for col in metadata_cols:
            if col in df.columns:
                result_df[col] = df[col].iloc[0]
        return result_df
    return pd.DataFrame()


def load_all_statements(data_folder: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Load all financial statements from Excel files."""
    excel_files = glob.glob(os.path.join(data_folder, '*.xlsx'))

    # Parse file info
    file_info_list = [info for file in excel_files if (info := parse_filename(file))]
    file_info_list.sort(key=lambda x: (x['year'], x['quarter'] or 'ZZZ'))

    print(f"Found {len(file_info_list)} Excel files")

    # Sheet name mappings
    sheet_mapping = {
        'income': {
            'primary': 'INCOME STATEMENTS',
            'alternative': 'INCOME_STATEMENTS',
            'special_2014': 'Income_Statements'
        },
        'balance': {
            'primary': 'BALANCE SHEETS',
            'alternative': 'BALANCE_SHEETS',
            'special_2014': 'Balance_Sheets'
        },
        'cashflow': {
            'primary': 'CASH FLOWS STATEMENTS',
            'alternative': 'CASH_FLOWS_STATEMENTS',
            'special_2014': 'Cash_Flows_Statements'
        }
    }

    income_statements = []
    balance_sheets = []
    cash_flow_statements = []

    print("Loading financial statements...")
    for file_info in file_info_list:
        file_path = file_info['filename']

        # Check for special 2014 Q1-Q3 format
        is_special_2014 = (
            file_info['year'] == 2014 and
            file_info['quarter'] in ['Q1', 'Q2', 'Q3'] and
            file_info['filing_type'] == '10Q'
        )

        # Load each statement type
        for stmt_type, sheets in sheet_mapping.items():
            primary = sheets['special_2014'] if is_special_2014 else sheets['primary']
            alternative = sheets['alternative'] if not is_special_2014 else sheets['primary']

            df = load_financial_statement(file_path, primary, alternative)
            if df is not None:
                processed_df = process_financial_dataframe(df, file_info)
                if processed_df is not None:
                    if stmt_type == 'income':
                        income_statements.append(processed_df)
                    elif stmt_type == 'balance':
                        balance_sheets.append(processed_df)
                    else:
                        cash_flow_statements.append(processed_df)

    print(f"\nLoaded:")
    print(f"  Income Statements: {len(income_statements)}")
    print(f"  Balance Sheets: {len(balance_sheets)}")
    print(f"  Cash Flow Statements: {len(cash_flow_statements)}")

    return income_statements, balance_sheets, cash_flow_statements

# Extract metrics from all statements
def extract_all_metrics(statements: List[pd.DataFrame], extract_func, statement_type: str) -> List[pd.DataFrame]:
    extracted = []
    for i, df in enumerate(statements):
        try:
            result = extract_func(df)
            if not result.empty:
                result['Statement_Type'] = statement_type
                extracted.append(result)
        except Exception as e:
            print(f"Error in {statement_type} {i}: {e}")

    print(f"Extracted metrics from {len(extracted)} {statement_type.lower()}s")
    return extracted

# Helper function to convert data frames to long format
def convert_to_long_format(extracted_list: List[pd.DataFrame]) -> pd.DataFrame:
    all_long_data = []

    metadata_cols = ['Metric', 'source_file', 'year', 'quarter', 'period', 'filing_type', 'Statement_Type']

    for df in extracted_list:
        value_cols = [col for col in df.columns if col not in metadata_cols]

        for val_col in value_cols:
            subset = df[metadata_cols + [val_col]].copy()
            subset['Period_Column'] = val_col
            subset['Value'] = pd.to_numeric(subset[val_col], errors='coerce')
            subset = subset.drop(columns=[val_col])
            subset = subset[subset['Value'].notna()]

            if not subset.empty:
                all_long_data.append(subset)

    return pd.concat(all_long_data, ignore_index=True) if all_long_data else pd.DataFrame()

#Q4 values do not exist in the data, so we need to calculate them
def calculate_q4_values(annual_data: pd.DataFrame, quarterly_data: pd.DataFrame) -> pd.DataFrame:
    q4_records = []

    for (year, metric, stmt_type), annual_group in annual_data.groupby(['year', 'Metric', 'Statement_Type']):
        annual_value = annual_group['Value'].iloc[0]

        if stmt_type == 'Balance Sheet':
            # For Balance Sheet, Q4 = Annual value directly
            q4_records.append({
                'Date': create_fiscal_date(year, 'Q4'),
                'Metric': metric,
                'Value': annual_value,
                'Statement_Type': stmt_type,
                'quarter': 'Q4',
                'year': year
            })
        else:
            # Special handling for cash balance metrics in Cash Flow statement
            if metric in ['Cash and Cash Equivalents - Beginning', 'Cash and Cash Equivalents - End']:
                if metric == 'Cash and Cash Equivalents - End':
                    # End of period for Q4 = Annual value (as is)
                    q4_records.append({
                        'Date': create_fiscal_date(year, 'Q4'),
                        'Metric': metric,
                        'Value': annual_value,
                        'Statement_Type': stmt_type,
                        'quarter': 'Q4',
                        'year': year
                    })
                else:  # Cash and Cash Equivalents - Beginning
                    # Beginning of Q4 = End of Q3
                    q3_end = quarterly_data[
                        (quarterly_data['year'] == year) &
                        (quarterly_data['quarter'] == 'Q3') &
                        (quarterly_data['Metric'] == 'Cash and Cash Equivalents - End')
                    ]
                    if not q3_end.empty:
                        q4_records.append({
                            'Date': create_fiscal_date(year, 'Q4'),
                            'Metric': metric,
                            'Value': q3_end['Value'].iloc[0],
                            'Statement_Type': stmt_type,
                            'quarter': 'Q4',
                            'year': year
                        })
            else:
                # For other Income Statement and Cash Flow metrics, calculate Q4 = Annual - (Q1 + Q2 + Q3)
                quarters = ['Q1', 'Q2', 'Q3']
                quarter_values = []

                for q in quarters:
                    q_data = quarterly_data[
                        (quarterly_data['year'] == year) &
                        (quarterly_data['quarter'] == q) &
                        (quarterly_data['Metric'] == metric)
                    ]
                    if not q_data.empty:
                        quarter_values.append(q_data['Value'].iloc[0])
                    else:
                        break

                if len(quarter_values) == 3:
                    q4_value = annual_value - sum(quarter_values)
                    q4_records.append({
                        'Date': create_fiscal_date(year, 'Q4'),
                        'Metric': metric,
                        'Value': q4_value,
                        'Statement_Type': stmt_type,
                        'quarter': 'Q4',
                        'year': year
                    })

    return pd.DataFrame(q4_records)

# Now that we have core metrics, helper function to calculate derived metrics
def calculate_derived_metrics(pivot_data: pd.DataFrame) -> pd.DataFrame:
    calculated_metrics = []

    metric_calculations = [
        # (Result Metric, Required Metrics, Calculation Function)
        ('EBITDA', ['Operating Income', 'Depreciation, Amortization, and Other'],
         lambda r: r['Operating Income'] + r['Depreciation, Amortization, and Other']),

        ('Free Cash Flow to Firm', ['Net Cash from Operations', 'Purchases of Property and Equipment'],
         lambda r: r['Net Cash from Operations'] - abs(r['Purchases of Property and Equipment'])),

        ('Gross Margin Percentage', ['Gross Margin', 'Revenue'],
         lambda r: (r['Gross Margin'] / r['Revenue'] * 100) if r['Revenue'] != 0 else None),

        ('Operating Margin Percentage', ['Operating Income', 'Revenue'],
         lambda r: (r['Operating Income'] / r['Revenue'] * 100) if r['Revenue'] != 0 else None),

        ('Net Profit Margin Percentage', ['Net Income', 'Revenue'],
         lambda r: (r['Net Income'] / r['Revenue'] * 100) if r['Revenue'] != 0 else None),

        ('Working Capital', ['Total Current Assets', 'Total Current Liabilities'],
         lambda r: r['Total Current Assets'] - r['Total Current Liabilities']),

        ('Current Ratio', ['Total Current Assets', 'Total Current Liabilities'],
         lambda r: (r['Total Current Assets'] / r['Total Current Liabilities']) if r['Total Current Liabilities'] != 0 else None),

        ('Debt-to-Equity Ratio', ['Total Liabilities', 'Total Stockholders Equity'],
         lambda r: (r['Total Liabilities'] / r['Total Stockholders Equity']) if r['Total Stockholders Equity'] != 0 else None),
    ]

    for _, row in pivot_data.iterrows():
        for metric_name, required_cols, calc_func in metric_calculations:
            if all(pd.notna(row.get(col)) for col in required_cols):
                value = calc_func(row)
                if value is not None:
                    calculated_metrics.append({
                        'Date': row['Date'],
                        'Metric': metric_name,
                        'Value': value
                    })

    return pd.DataFrame(calculated_metrics)

# ROA ROE calculations
def calculate_roa_roe(pivot_data: pd.DataFrame) -> pd.DataFrame:
    pivot_data_sorted = pivot_data.sort_values('Date').copy()
    metrics = []

    for i in range(len(pivot_data_sorted)):
        row = pivot_data_sorted.iloc[i]
        date = row['Date']
        net_income = row.get('Net Income')

        if pd.notna(net_income):
            # ROA calculation
            total_assets = row.get('Total Assets')
            if pd.notna(total_assets):
                if i > 0:
                    prev_assets = pivot_data_sorted.iloc[i-1].get('Total Assets')
                    if pd.notna(prev_assets):
                        avg_assets = (total_assets + prev_assets) / 2
                    else:
                        avg_assets = total_assets
                else:
                    avg_assets = total_assets

                if avg_assets != 0:
                    metrics.append({
                        'Date': date,
                        'Metric': 'Return on Assets (ROA)',
                        'Value': (net_income / avg_assets) * 100
                    })

            # ROE calculation
            stockholders_equity = row.get('Total Stockholders Equity')
            if pd.notna(stockholders_equity):
                if i > 0:
                    prev_equity = pivot_data_sorted.iloc[i-1].get('Total Stockholders Equity')
                    if pd.notna(prev_equity):
                        avg_equity = (stockholders_equity + prev_equity) / 2
                    else:
                        avg_equity = stockholders_equity
                else:
                    avg_equity = stockholders_equity

                if avg_equity != 0:
                    metrics.append({
                        'Date': date,
                        'Metric': 'Return on Equity (ROE)',
                        'Value': (net_income / avg_equity) * 100
                    })

    return pd.DataFrame(metrics)

# Main function to help with extraction for Excel model
def create_wide_format(df: pd.DataFrame, statement_type: str, metric_order: List[str]) -> pd.DataFrame:
    filtered_df = df[df['Statement_Type'] == statement_type].copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    wide_df = filtered_df.pivot_table(
        index='Metric',
        columns='Date',
        values='Value',
        aggfunc='first'
    )

    # Reorder metrics
    ordered_metrics = [m for m in metric_order if m in wide_df.index]
    other_metrics = [m for m in wide_df.index if m not in ordered_metrics]
    wide_df = wide_df.reindex(ordered_metrics + other_metrics)

    return wide_df.sort_index(axis=1)

# Processing function
def main():
    print("Microsoft Financial Data Processing")
    print("=" * 60)

    # Configuration
    data_folder = 'data/raw data'
    output_folder = 'data/processed'
    os.makedirs(output_folder, exist_ok=True)

    # Load all statements
    print("\n1. Loading financial statements...")
    income_statements, balance_sheets, cash_flow_statements = load_all_statements(data_folder)

    # Extract metrics
    print("\n2. Extracting metrics...")
    income_extracted = extract_all_metrics(income_statements, extract_income_metrics, 'Income Statement')
    balance_extracted = extract_all_metrics(balance_sheets, extract_balance_metrics, 'Balance Sheet')
    cashflow_extracted = extract_all_metrics(cash_flow_statements, extract_cashflow_metrics, 'Cash Flow')

    # Convert to long format
    print("\n3. Converting to long format...")
    income_long = convert_to_long_format(income_extracted)
    balance_long = convert_to_long_format(balance_extracted)
    cashflow_long = convert_to_long_format(cashflow_extracted)

    all_metrics = pd.concat([income_long, balance_long, cashflow_long], ignore_index=True)
    print(f"Total data points: {len(all_metrics)}")

    # Create master data with fixed dates
    print("\n4. Creating master data with fixed dates...")
    master_data = all_metrics.copy()
    master_data['Date'] = master_data.apply(
        lambda row: create_fiscal_date(row['year'], row['quarter']), axis=1
    )

    master_data = master_data[['Date', 'Metric', 'Value', 'Statement_Type', 'quarter', 'year']].copy()
    master_data = master_data.drop_duplicates(subset=['Date', 'Metric'], keep='first')
    master_data = master_data.sort_values(['Date', 'Statement_Type', 'Metric'])

    # Calculate Q4 values
    print("\n5. Calculating Q4 values...")
    annual_data = master_data[master_data['quarter'].isna()].copy()
    quarterly_data = master_data[master_data['quarter'].notna()].copy()

    q4_data = calculate_q4_values(annual_data, quarterly_data)

    final_master_data = pd.concat([
        quarterly_data[quarterly_data['quarter'] != 'Q4'],
        q4_data
    ], ignore_index=True).sort_values(['Date', 'Statement_Type', 'Metric'])

    # Calculate derived metrics
    print("\n6. Calculating derived metrics...")
    pivot_data = final_master_data.pivot_table(
        index=['Date', 'year', 'quarter'],
        columns='Metric',
        values='Value',
        aggfunc='first'
    ).reset_index()

    derived_metrics = calculate_derived_metrics(pivot_data)
    roa_roe_metrics = calculate_roa_roe(pivot_data)
    all_derived_metrics = pd.concat([derived_metrics, roa_roe_metrics], ignore_index=True)

    # Create final master data
    print("\n7. Creating final master data...")
    original_metrics = final_master_data[['Date', 'Metric', 'Value']].copy()
    master_long_data = pd.concat([original_metrics, all_derived_metrics], ignore_index=True)
    master_long_data = master_long_data.drop_duplicates(subset=['Date', 'Metric'], keep='first')
    master_long_data = master_long_data.sort_values(['Date', 'Metric'])

    # Save master long data
    output_csv = os.path.join(output_folder, 'microsoft_master_long_data.csv')
    master_long_data.to_csv(output_csv, index=False)
    print(f"\nSaved master long data to: {output_csv}")

    # Create Excel export
    print("\n8. Creating wide format Excel export...")

    # Add Statement_Type back for Excel export
    statement_types = final_master_data[['Date', 'Metric', 'Statement_Type']].drop_duplicates()
    master_with_types = master_long_data.merge(statement_types, on=['Date', 'Metric'], how='left')

    # Assign statement types for calculated metrics
    calculated_metric_types = {
        'EBITDA': 'Income Statement',
        'Free Cash Flow to Firm': 'Cash Flow',
        'Gross Margin Percentage': 'Income Statement',
        'Operating Margin Percentage': 'Income Statement',
        'Net Profit Margin Percentage': 'Income Statement',
        'Working Capital': 'Balance Sheet',
        'Current Ratio': 'Balance Sheet',
        'Debt-to-Equity Ratio': 'Balance Sheet',
        'Return on Assets (ROA)': 'Balance Sheet',
        'Return on Equity (ROE)': 'Balance Sheet'
    }

    for metric, stmt_type in calculated_metric_types.items():
        mask = (master_with_types['Metric'] == metric) & (master_with_types['Statement_Type'].isna())
        master_with_types.loc[mask, 'Statement_Type'] = stmt_type

    # Define metric orders
    income_metrics_order = [
        'Revenue', 'Cost of Revenue', 'Gross Margin', 'Gross Margin Percentage',
        'Research and Development', 'Sales and Marketing', 'General and Administrative',
        'Operating Income', 'Operating Margin Percentage', 'EBITDA',
        'Other Income (Expense), Net', 'Income Before Income Taxes',
        'Provision for Income Taxes', 'Net Income', 'Net Profit Margin Percentage',
        'Basic Earnings Per Share', 'Diluted Earnings Per Share',
        'Weighted-Average Shares Outstanding - Basic', 'Weighted-Average Shares Outstanding - Diluted'
    ]

    balance_metrics_order = [
        'Cash and Cash Equivalents', 'Short-Term Investments', 'Accounts Receivable, Net',
        'Inventories', 'Total Current Assets', 'Property and Equipment, Net',
        'Goodwill', 'Intangible Assets, Net', 'Total Assets',
        'Accounts Payable', 'Accrued Compensation', 'Unearned Revenue',
        'Total Current Liabilities', 'Long-Term Debt', 'Total Liabilities',
        'Common Stock', 'Retained Earnings', 'Total Stockholders Equity',
        'Working Capital', 'Current Ratio', 'Debt-to-Equity Ratio',
        'Return on Assets (ROA)', 'Return on Equity (ROE)'
    ]

    cashflow_metrics_order = [
        'Net Income', 'Depreciation, Amortization, and Other',
        'Change in Accounts Receivable', 'Change in Inventories', 'Change in Accounts Payable',
        'Net Cash from Operations', 'Purchases of Property and Equipment',
        'Net Cash Used in Investing', 'Dividends Paid',
        'Repurchases of Common Stock', 'Net Cash Used in Financing',
        'Effect of Foreign Exchange Rates', 'Net Change in Cash and Cash Equivalents',
        'Cash and Cash Equivalents - Beginning', 'Cash and Cash Equivalents - End',
        'Free Cash Flow to Firm'
    ]

    # Create wide format dataframes
    income_wide = create_wide_format(master_with_types, 'Income Statement', income_metrics_order)
    balance_wide = create_wide_format(master_with_types, 'Balance Sheet', balance_metrics_order)
    cashflow_wide = create_wide_format(master_with_types, 'Cash Flow', cashflow_metrics_order)

    # Export to Excel
    output_excel = os.path.join(output_folder, 'microsoft_historical_financials_wide.xlsx')
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        income_wide.to_excel(writer, sheet_name='Income Statement (Hist)')
        balance_wide.to_excel(writer, sheet_name='Balance Sheet (Hist)')
        cashflow_wide.to_excel(writer, sheet_name='Cash Flow (Hist)')

        workbook = writer.book
        num_format = workbook.add_format({'num_format': '#,##0;(#,##0)'})

        for sheet_name in ['Income Statement (Hist)', 'Balance Sheet (Hist)', 'Cash Flow (Hist)']:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:A', 40)
            worksheet.set_column('B:ZZ', 15, num_format)
            worksheet.freeze_panes(1, 1)

    print(f"Financial data exported to: {output_excel}")


if __name__ == "__main__":
    main()
