import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up the page
st.set_page_config(
    page_title="Indian Stock Financial Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Indian Stock Financial Analysis Tool")
st.markdown("""
This tool fetches the latest 10 quarters of financial data for any Indian stock and provides AI-driven analysis using Gemini API.
Enter a stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS) to get started.
""")

# Initialize session state for API key and data
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'company_name' not in st.session_state:
    st.session_state.company_name = ''

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key:", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key
        st.success("API key saved!")
    
    st.markdown("---")
    st.info("""
    **Note:** 
    - Indian stocks typically have the '.NS' suffix (NSE)
    - You need a Gemini API key for the analysis feature
    - Example symbols: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, SBIN.NS
    """)

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get company name
        company_name = info.get('longName', symbol)
        
        # Get financial data
        financials = stock.quarterly_financials
        balance_sheet = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        
        if financials.empty:
            return None, "No quarterly financial data available for this stock."
        
        # Get the last 10 quarters
        quarters = financials.columns[:10]
        
        # Extract relevant metrics
        financial_data = {}
        
        # Sales (Total Revenue)
        if 'Total Revenue' in financials.index:
            financial_data['Sales'] = financials.loc['Total Revenue', quarters].values[:10] / 10000000  # Convert to Crores
        else:
            return None, "Sales data not available for this stock."
        
        # Operating Profit (Operating Income)
        if 'Operating Income' in financials.index:
            financial_data['Operating Profit'] = financials.loc['Operating Income', quarters].values[:10] / 10000000  # Convert to Crores
        else:
            # Calculate operating profit if not directly available
            if 'Total Revenue' in financials.index and 'Total Operating Expenses' in financials.index:
                revenue = financials.loc['Total Revenue', quarters].values[:10]
                opex = financials.loc['Total Operating Expenses', quarters].values[:10]
                financial_data['Operating Profit'] = (revenue - opex) / 10000000  # Convert to Crores
            else:
                financial_data['Operating Profit'] = [0] * 10
        
        # Calculate OPM%
        financial_data['OPM%'] = [
            (op / sales) * 100 if sales != 0 else 0 
            for op, sales in zip(financial_data['Operating Profit'], financial_data['Sales'])
        ]
        
        # Net Profit (Net Income)
        if 'Net Income' in financials.index:
            financial_data['Net Profit'] = financials.loc['Net Income', quarters].values[:10] / 10000000  # Convert to Crores
        else:
            financial_data['Net Profit'] = [0] * 10
        
        # EPS (Earnings Per Share)
        if 'Basic EPS' in financials.index:
            financial_data['EPS'] = financials.loc['Basic EPS', quarters].values[:10]
        else:
            # Calculate EPS from net income and shares outstanding
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding and 'Net Profit' in financial_data:
                # Convert net profit from crores to actual amount and divide by shares
                financial_data['EPS'] = [
                    (net_income * 10000000) / shares_outstanding 
                    for net_income in financial_data['Net Profit']
                ]
            else:
                financial_data['EPS'] = [0] * 10
        
        # Format quarter names for display
        quarter_names = [q.strftime('%Y-Q%q') for q in quarters]
        
        # Create DataFrame
        df = pd.DataFrame(financial_data, index=quarter_names)
        df.index.name = 'Quarter'
        
        # Reverse to show oldest first
        df = df.iloc[::-1]
        
        return df, company_name
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Function to generate analysis using Gemini API
def generate_analysis(company_name, financial_data):
    if not st.session_state.api_key:
        return "Please enter your Gemini API key in the sidebar to generate analysis."
    
    try:
        # Configure Gemini API
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare the prompt
        prompt = f"""
        Analyze the financial performance of {company_name} based on the following quarterly data (values in Crores INR, except EPS):
        
        {financial_data.to_string()}
        
        Please provide a comprehensive analysis covering:
        1. Sales trend and growth pattern
        2. Operating profit margin (OPM%) trajectory and what it indicates
        3. Net profit performance and its relation to operating profit
        4. EPS growth and what it means for investors
        5. Overall financial health and future outlook based on these trends
        
        Keep the analysis professional yet accessible for retail investors.
        Highlight any concerning trends or positive indicators.
        Provide specific insights about each financial metric.
        """
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}. Please check your API key and try again."

# Function to create visualizations
def create_visualizations(financial_data, company_name):
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Sales Trend (₹ Crores)', 'Operating Profit (₹ Crores)', 
                       'OPM%', 'Net Profit (₹ Crores)', 'EPS'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    quarters = financial_data.index
    
    # Sales trend
    fig.add_trace(
        go.Scatter(x=quarters, y=financial_data['Sales'], name='Sales', 
                  line=dict(color='blue', width=3), marker=dict(size=8)),
        row=1, col=1
    )
    
    # Operating Profit
    fig.add_trace(
        go.Scatter(x=quarters, y=financial_data['Operating Profit'], name='Operating Profit', 
                  line=dict(color='green', width=3), marker=dict(size=8)),
        row=1, col=2
    )
    
    # OPM%
    fig.add_trace(
        go.Scatter(x=quarters, y=financial_data['OPM%'], name='OPM%', 
                  line=dict(color='red', width=3), marker=dict(size=8)),
        row=2, col=1
    )
    
    # Net Profit
    fig.add_trace(
        go.Scatter(x=quarters, y=financial_data['Net Profit'], name='Net Profit', 
                  line=dict(color='purple', width=3), marker=dict(size=8)),
        row=2, col=2
    )
    
    # EPS
    fig.add_trace(
        go.Scatter(x=quarters, y=financial_data['EPS'], name='EPS', 
                  line=dict(color='orange', width=3), marker=dict(size=8)),
        row=3, col=1
    )
    
    # Hide empty subplot
    fig.add_trace(go.Scatter(x=[None], y=[None], showlegend=False), row=3, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{company_name} - Financial Performance Trends (Last 10 Quarters)",
        showlegend=True,
        template="plotly_white"
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="₹ Crores", row=1, col=1)
    fig.update_yaxes(title_text="₹ Crores", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    fig.update_yaxes(title_text="₹ Crores", row=2, col=2)
    fig.update_yaxes(title_text="Earnings per Share", row=3, col=1)
    
    return fig

# Function to calculate growth metrics
def calculate_growth_metrics(financial_data):
    growth_data = {}
    
    # Calculate quarter-over-quarter growth
    for column in financial_data.columns:
        if column != 'OPM%':  # OPM% is already a percentage
            growth_data[f'{column} QoQ Growth'] = financial_data[column].pct_change() * 100
        else:
            growth_data[f'{column} Change'] = financial_data[column].diff()
    
    growth_df = pd.DataFrame(growth_data, index=financial_data.index)
    growth_df = growth_df.round(2)
    
    return growth_df

# Main app
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("Enter Indian Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("Analyze Stock", type="primary")

if analyze_btn or st.session_state.financial_data is not None:
    if symbol:
        with st.spinner("Fetching financial data..."):
            financial_data, company_name = fetch_stock_data(symbol)
        
        if financial_data is not None:
            st.session_state.financial_data = financial_data
            st.session_state.company_name = company_name
            
            st.success(f"Financial data retrieved for {company_name}!")
            
            # Display the financial data
            st.subheader("Financial Data (Last 10 Quarters)")
            display_df = financial_data.copy()
            display_df['Sales'] = display_df['Sales'].apply(lambda x: f"₹ {x:,.2f} Cr")
            display_df['Operating Profit'] = display_df['Operating Profit'].apply(lambda x: f"₹ {x:,.2f} Cr")
            display_df['OPM%'] = display_df['OPM%'].apply(lambda x: f"{x:.2f}%")
            display_df['Net Profit'] = display_df['Net Profit'].apply(lambda x: f"₹ {x:,.2f} Cr")
            display_df['EPS'] = display_df['EPS'].apply(lambda x: f"₹ {x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Calculate and display growth metrics
            st.subheader("Growth Metrics")
            growth_df = calculate_growth_metrics(financial_data)
            st.dataframe(growth_df.style.format("{:.2f}%"), use_container_width=True)
            
            # Create and display visualizations
            st.subheader("Financial Trends")
            fig = create_visualizations(financial_data, company_name)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display analysis
            st.subheader("AI-Powered Financial Analysis")
            if st.button("Generate Analysis", type="secondary"):
                with st.spinner("Generating analysis using Gemini AI..."):
                    analysis = generate_analysis(company_name, financial_data)
                
                st.write(analysis)
            
        else:
            st.error(company_name)  # In this case, company_name contains the error message
    else:
        st.warning("Please enter a stock symbol.")

# Add some information about the app
st.markdown("---")
st.markdown("""
### How to Use This Tool:
1. Enter an Indian stock symbol with the `.NS` suffix (for NSE-listed companies)
2. Click the "Analyze Stock" button to fetch financial data
3. View the financial metrics and visualizations
4. Get AI-powered analysis by providing your Gemini API key and clicking "Generate Analysis"

### Popular Indian Stock Symbols:
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- SBIN.NS (State Bank of India)
- ICICIBANK.NS (ICICI Bank)
- HINDUNILVR.NS (Hindustan Unilever)
- BAJFINANCE.NS (Bajaj Finance)

### Data Source:
Financial data is sourced from Yahoo Finance using the yfinance library.

### Note:
This tool is for educational purposes only. Always consult with a financial advisor before making investment decisions.
""")
