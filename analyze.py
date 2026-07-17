import yfinance as yf
import pandas as pd
from joblib import Memory
import shutil
import os
import numpy as np
from google import genai

# =====================================================================
# CONFIGURATION
# =====================================================================
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if os.path.exists("./yf_cache"):
    shutil.rmtree("./yf_cache")

memory = Memory("./yf_cache", verbose=0)

@memory.cache
def fetch_financial_data(ticker_symbol):
    print(f"--- Fetching fresh data for {ticker_symbol}... ---")
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        return {
            "financials": ticker.quarterly_financials,
            "cashflow": ticker.quarterly_cashflow,
            "balance_sheet": ticker.quarterly_balance_sheet,
            "trailing_pe": info.get("trailingPE", "N/A"),
            "market_cap": info.get("marketCap", "N/A")
        }
    except Exception:
        return {"financials": pd.DataFrame(), "cashflow": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "trailing_pe": "N/A", "market_cap": "N/A"}

# --- DATA PROCESSING ---
tickers = ['MCD']
table_rows = []
statement_order = ['Income Statement', 'Balance Sheet', 'Cash Flow']
metric_order = ['Period Ending', 'Market Cap', 'Total Revenue', 'Other Income Expense', 'Interest Expense', 'Net Income', 'Basic EPS', 'Price to Earnings', 'Basic Average Shares', 'Beginning Cash Position', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'End Cash Position', 'Free Cash Flow', 'Total Assets', 'Total Liabilities Net Minority Interest', 'Total Debt', 'Total Equity Gross Minority Interest']

for j in tickers:
    cached_data = fetch_financial_data(j)
    income_statement = ['Period Ending', 'Market Cap', 'Total Revenue', 'Other Income Expense', 'Interest Expense', 'Net Income', 'Basic EPS', 'Price to Earnings', 'Basic Average Shares']
    cash_flow = ['Period Ending', 'End Cash Position', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Beginning Cash Position', 'Free Cash Flow']
    balance_sheet = ['Period Ending', 'Total Assets', 'Total Liabilities Net Minority Interest', 'Total Debt', 'Total Equity Gross Minority Interest']
    
    for i in [income_statement, cash_flow, balance_sheet]:
        if i is income_statement: df, statement_type = cached_data["financials"], "Income Statement"
        elif i is cash_flow: df, statement_type = cached_data["cashflow"], "Cash Flow"
        else: df, statement_type = cached_data["balance_sheet"], "Balance Sheet"
            
        latest_date = df.columns[0]
        for metric in i:
            if metric == 'Period Ending': formatted_val = latest_date.strftime('%Y-%m-%d')
            elif metric == 'Market Cap':
                mc_val = cached_data.get("market_cap")
                formatted_val = f"${(mc_val / 1_000_000_000):,.3f}B" if isinstance(mc_val, (int, float)) else "N/A"
            elif metric == 'Price to Earnings':
                pe_val = cached_data.get("trailing_pe")
                formatted_val = f"{pe_val:.2f}x" if isinstance(pe_val, (int, float)) else "N/A"
            else:
                try:
                    raw_val = df.loc[metric, latest_date]
                    if pd.isna(raw_val): formatted_val = "N/A"
                    elif metric == "Basic EPS": formatted_val = f"${raw_val:,.2f}"
                    else: formatted_val = f"${(raw_val / 1_000_000_000):,.3f}B"
                except KeyError: formatted_val = "N/A"
            table_rows.append({"Ticker": j, "Statement Type": statement_type, "Metric": metric, "Value": formatted_val})

df_pivoted = pd.DataFrame(table_rows).pivot(index=["Statement Type", "Metric"], columns="Ticker", values="Value").reset_index()

# --- GEMINI ANALYSIS & REPORT GENERATION ---
def run_pipeline(df):
    # 1. Ask Gemini
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"Analyze this financial data and provide insights:\n{df.to_string()}"
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    
    # 2. Save as HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Financial Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        .gemini-box {{ background: #f9f9f9; padding: 20px; border-left: 5px solid #4285f4; margin-top: 20px; }}
    </style></head>
    <body>
        <a href="index.html">← Back to Home</a>
        <h1>Quarterly Financial Analysis</h1>
        {df.to_html(index=False)}
        <h2>Gemini Insights</h2>
        <div class="gemini-box">{response.text.replace(chr(10), '<br>')}</div>
    </body>
    </html>
    """
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Report generated successfully.")

run_pipeline(df_pivoted)
