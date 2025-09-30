import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json
import hashlib
from typing import Dict, List, Tuple, Any

# Import insights generator (will be used after DataProcessor is defined)
# from insights_generator import InsightsGenerator

# ML and AI imports
try:
    import openai
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import chromadb
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("Some AI/ML packages not available. Install requirements.txt for full functionality.")

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Tipsy Topsy Analytics Dashboard",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .file-upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f1f3f4;
        color: #333;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def format_indian_number(num):
    """Format number in Indian numbering system with commas"""
    if num < 0:
        return '-' + format_indian_number(-num)
    
    s = str(int(num))
    if len(s) <= 3:
        return s
    
    # Last 3 digits
    result = s[-3:]
    s = s[:-3]
    
    # Add commas every 2 digits from right to left
    while len(s) > 0:
        if len(s) <= 2:
            result = s + ',' + result
            break
        else:
            result = s[-2:] + ',' + result
            s = s[:-2]
    
    return result

class DataProcessor:
    """Unified data processor for vzrm_6.csv file"""

    def __init__(self):
        self.df = None
        self.sales_df = None
        self.returns_df = None
        self.processed_data = {}
        self.insights_cache = {}

    def load_and_process_data(self, file_path: str) -> bool:
        """Load and process the vzrm_6.csv file"""
        try:
            # Read the CSV file
            self.df = pd.read_csv(file_path)

            # Clean column names
            self.df.columns = self.df.columns.str.strip()

            # Convert date column
            self.df['Bill Date'] = pd.to_datetime(self.df['Bill Date'], format='%d/%m/%Y', errors='coerce')

            # Remove rows with invalid dates
            self.df = self.df[~self.df['Bill Date'].isna()]

            # Remove summary rows
            if 'Tran Type' in self.df.columns:
                self.df = self.df[~self.df['Tran Type'].isin(['*Sub Total*', '*Sub Total* - *30/01/2025*'])]

            # Convert numeric columns - handle the concatenated values properly
            numeric_columns = ['Cost', 'MRP', 'Doc Rate', 'Qty', 'Value', 'Tax', 'Item - Discount', 'Bill - Discount', 'Total - Discount']
            for col in numeric_columns:
                if col in self.df.columns:
                    # First convert to string to handle any weird formatting
                    self.df[col] = self.df[col].astype(str)
                    # Remove any non-numeric characters except decimal points and minus signs
                    self.df[col] = self.df[col].str.replace(r'[^\d.-]', '', regex=True)
                    # Convert to numeric
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

            # Properly separate sales and returns based on transaction type
            # This matches the logic from the working new.py file
            self.sales_df = self.df[self.df['Tran Type'] == 'Sales'].copy()
            self.returns_df = self.df[self.df['Tran Type'] == 'Return'].copy()

            # Clean return values (remove parentheses and make positive for calculations)
            if not self.returns_df.empty:
                # Remove parentheses and convert to positive values for calculations
                self.returns_df['Value'] = self.returns_df['Value'].astype(str).str.replace(r'[()]', '', regex=True).astype(float)
                # Make quantities positive for calculations
                self.returns_df['Qty'] = self.returns_df['Qty'].abs()

            # Add computed columns
            self._add_computed_columns()

            # Generate insights
            self._generate_insights()

            return True

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return False

    def _add_computed_columns(self):
        """Add computed columns for analysis"""
        # Brand-Product combination
        self.df['Brand_Product'] = self.df['Brand Code'] + ' - ' + self.df['Product Code']

        # Profit calculation (MRP - Cost)
        self.df['Profit'] = self.df['MRP'] - self.df['Cost']

        # Profit margin
        self.df['Profit_Margin'] = (self.df['Profit'] / self.df['MRP'] * 100).round(2)

        # Day of week
        self.df['Day_of_Week'] = self.df['Bill Date'].dt.day_name()

        # Month
        self.df['Month'] = self.df['Bill Date'].dt.month_name()

    def _generate_insights(self):
        """Generate various insights from the data"""
        if self.df is None or self.df.empty:
            return

        insights = {}

        # Basic metrics - use actual data from the working new.py logic
        insights['total_sales'] = len(self.sales_df) if self.sales_df is not None else 0
        insights['total_returns'] = len(self.returns_df) if self.returns_df is not None else 0
        insights['total_revenue'] = self.sales_df['Value'].sum() if self.sales_df is not None else 0
        insights['total_return_value'] = abs(self.returns_df['Value'].sum()) if self.returns_df is not None else 0
        insights['net_revenue'] = insights['total_revenue'] - insights['total_return_value']
        insights['unique_products'] = self.sales_df['Product Code'].nunique() if self.sales_df is not None else 0
        insights['unique_brands'] = self.sales_df['Brand Code'].nunique() if self.sales_df is not None else 0

        # Date range - use filtered sales data if available, otherwise use full dataset
        if self.sales_df is not None and not self.sales_df.empty:
            date_df = self.sales_df
        else:
            date_df = self.df

        insights['date_range'] = {
            'start': date_df['Bill Date'].min().strftime('%d/%m/%Y'),
            'end': date_df['Bill Date'].max().strftime('%d/%m/%Y')
        }

        # Top performing products
        product_performance = self.sales_df.groupby(['Brand Code', 'Product Code']).agg({
            'Qty': 'sum',
            'Value': 'sum'
        }).reset_index() if self.sales_df is not None else pd.DataFrame()

        insights['top_products_qty'] = product_performance.nlargest(10, 'Qty')[['Brand Code', 'Product Code', 'Qty']].to_dict('records')
        insights['top_products_value'] = product_performance.nlargest(10, 'Value')[['Brand Code', 'Product Code', 'Value']].to_dict('records')

        # Brand analysis
        brand_performance = self.sales_df.groupby('Brand Code').agg({
            'Qty': 'sum',
            'Value': 'sum',
            'Product Code': 'nunique'
        }).reset_index() if self.sales_df is not None else pd.DataFrame()

        insights['top_brands_qty'] = brand_performance.nlargest(10, 'Qty')[['Brand Code', 'Qty']].to_dict('records')
        insights['top_brands_value'] = brand_performance.nlargest(10, 'Value')[['Brand Code', 'Value']].to_dict('records')

        # Return analysis
        if self.returns_df is not None and not self.returns_df.empty:
            return_analysis = self.returns_df.groupby(['Brand Code', 'Product Code']).agg({
                'Qty': lambda x: abs(x.sum()),
                'Value': lambda x: abs(x.sum())
            }).reset_index()

            insights['return_analysis'] = return_analysis.to_dict('records')
            # Calculate return rate based on quantity, not value
            total_sales_qty = self.sales_df['Qty'].sum() if self.sales_df is not None else 0
            total_return_qty = abs(self.returns_df['Qty'].sum()) if self.returns_df is not None else 0
            insights['return_rate'] = (total_return_qty / total_sales_qty * 100) if total_sales_qty > 0 else 0
            insights['return_count'] = int(total_return_qty)  # Store the actual count
        else:
            insights['return_analysis'] = []
            insights['return_rate'] = 0
            insights['return_count'] = 0

        # Daily sales pattern
        daily_sales = self.sales_df.groupby('Bill Date').agg({
            'Qty': 'sum',
            'Value': 'sum'
        }).reset_index() if self.sales_df is not None else pd.DataFrame()

        insights['daily_sales_trend'] = daily_sales.to_dict('records')

        # Size distribution
        if 'Size' in self.sales_df.columns:
            size_analysis = self.sales_df.groupby('Size').agg({
                'Qty': 'sum'
            }).reset_index() if self.sales_df is not None else pd.DataFrame()

            insights['size_distribution'] = size_analysis.to_dict('records')
        else:
            insights['size_distribution'] = []

        self.insights_cache = insights

    def get_insights(self) -> Dict[str, Any]:
        """Get generated insights"""
        return self.insights_cache

# Import insights generator after DataProcessor is defined to avoid circular imports
from insights_generator import InsightsGenerator

class RAGSystem:
    """RAG system for data querying using OpenRouter API"""

    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.context = ""
        self.is_initialized = False

    def prepare_documents(self, data_processor):
        """Prepare context from processed data for RAG"""
        documents = []

        try:
            # Add basic statistics
            insights = data_processor.get_insights()
            basic_stats = f"""
            Business Overview:
            - Total Sales Transactions: {insights['total_sales']:,}
            - Total Returns: {insights['total_returns']:,}
            - Total Revenue: ₹{insights['total_revenue']:,.0f}
            - Net Revenue: ₹{insights['net_revenue']:,.0f}
            - Return Rate: {insights['return_rate']:.2f}% ({format_indian_number(insights.get('return_count', 0))} returns)
            - Unique Products: {insights['unique_products']}
            - Unique Brands: {insights['unique_brands']}
            - Date Range: {insights['date_range']['start']} to {insights['date_range']['end']}
            """
            documents.append(basic_stats)

            # Add top products information
            if insights['top_products_qty']:
                top_products_text = "Top 10 Products by Quantity Sold:\n"
                for i, product in enumerate(insights['top_products_qty'][:10], 1):
                    top_products_text += f"{i}. {product['Brand Code']} - {product['Product Code']}: {product['Qty']} units\n"
                documents.append(top_products_text)

            if insights['top_products_value']:
                top_products_value_text = "Top 10 Products by Revenue:\n"
                for i, product in enumerate(insights['top_products_value'][:10], 1):
                    top_products_value_text += f"{i}. {product['Brand Code']} - {product['Product Code']}: ₹{product['Value']:,.0f}\n"
                documents.append(top_products_value_text)

            # Add brand information
            if insights['top_brands_qty']:
                brands_text = "Top Brands by Quantity:\n"
                for i, brand in enumerate(insights['top_brands_qty'][:10], 1):
                    brands_text += f"{i}. {brand['Brand Code']}: {brand['Qty']} units\n"
                documents.append(brands_text)

            # Add return analysis with size details
            if insights['return_analysis']:
                returns_text = "Return Analysis (Aggregated by Brand-Product):\n"
                for ret in insights['return_analysis'][:15]:
                    returns_text += f"{ret['Brand Code']} - {ret['Product Code']}: {ret['Qty']} returns, ₹{ret['Value']:,.0f} value\n"
                documents.append(returns_text)

                # Add detailed return transactions with sizes (limit to recent ones for context size)
                detailed_returns_text = "\nDetailed Return Transactions (with sizes):\n"
                if data_processor.returns_df is not None and not data_processor.returns_df.empty:
                    # Get returns with size information, limit to most recent for context efficiency
                    returns_with_sizes = data_processor.returns_df[['Brand Code', 'Product Code', 'Size', 'Qty', 'Value']].head(20)
                    for _, ret in returns_with_sizes.iterrows():
                        size_info = f" (Size: {ret['Size']})" if pd.notna(ret['Size']) and ret['Size'] != '' else ""
                        detailed_returns_text += f"{ret['Brand Code']} - {ret['Product Code']}{size_info}: {abs(ret['Qty'])} return(s), ₹{abs(ret['Value']):,.0f} value\n"

                    documents.append(detailed_returns_text)

            # Add sample sales transactions with sizes (for context about available sizes)
            if data_processor.sales_df is not None and not data_processor.sales_df.empty:
                sales_sizes_text = "\nSample Sales Transactions (showing available sizes):\n"
                # Get diverse sample of sales with sizes
                sales_sample = data_processor.sales_df[['Brand Code', 'Product Code', 'Size']].dropna().head(15)
                for _, sale in sales_sample.iterrows():
                    if pd.notna(sale['Size']) and sale['Size'] != '':
                        sales_sizes_text += f"{sale['Brand Code']} - {sale['Product Code']}: Size {sale['Size']}\n"
                documents.append(sales_sizes_text)

            # Add daily sales trend
            if insights['daily_sales_trend']:
                daily_text = "Daily Sales Trend:\n"
                for day in insights['daily_sales_trend'][:7]:  # Last 7 days
                    daily_text += f"{day['Bill Date'].strftime('%d/%m/%Y')}: {day['Qty']} units, ₹{day['Value']:,.0f}\n"
                documents.append(daily_text)

            self.context = "\n\n".join(documents)
            self.is_initialized = True
            return self.context

        except Exception as e:
            st.error(f"Error preparing documents: {str(e)}")
            return ""

    def initialize_rag(self, context: str):
        """Initialize the RAG system with context"""
        self.context = context
        self.is_initialized = True

    def query(self, question: str) -> str:
        """Query the RAG system using OpenRouter API directly"""
        if not self.api_key:
            return "RAG system not available. Please check your OpenRouter API key."

        if not self.is_initialized:
            return "RAG system not initialized. Please prepare documents first."

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tipsy-topsy-dashboard.com",
                "X-Title": "Tipsy Topsy Analytics Dashboard"
            }

            # Include context in the system message with STRICT anti-hallucination instructions
            system_message = f"""You are a STRICT data analysis assistant for a clothing store dashboard. You MUST ONLY use the provided context data to answer questions. NEVER generate, invent, or hallucinate any data that is not explicitly present in the context.

CRITICAL RULES:
1. ONLY answer based on the data provided in the context below
2. If information is not in the context, say "This information is not available in the current data"
3. NEVER make assumptions or estimates about sizes, quantities, or any missing data
4. NEVER generate fake numbers, trends, or insights
5. NEVER respond to questions outside of clothing store sales dashboard topics
6. For size information: Use the detailed transaction data provided in the context
7. For return counts: Use both aggregated return analysis and detailed transactions
8. ONLY provide insights and analysis based on the actual data shown

AVAILABLE DATA IN CONTEXT:
- Business overview with sales/returns counts and revenue figures
- Top products by quantity and revenue (aggregated)
- Top brands by quantity (aggregated)
- Return analysis (aggregated by brand-product combinations)
- Detailed return transactions with sizes (individual transactions)
- Sample sales transactions showing available sizes
- Daily sales trends

Context Data (ONLY use this - no external knowledge):
{self.context}

Answer ONLY about the clothing store dashboard data above. If specific details like exact sizes for particular returns are needed, check the detailed transaction sections."""

            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content']
                    return answer
                else:
                    return "Unable to get response from API - no choices in response"
            else:
                return f"API Error {response.status_code}: {response.text}"

        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

    def get_context(self) -> str:
        """Get the current context data"""
        return self.context

    def generate_sales_summary(self) -> str:
        """Generate a comprehensive sales summary using GPT-4o"""
        if not self.api_key:
            return "RAG system not available. Please check your OpenRouter API key."

        if not self.is_initialized:
            return "RAG system not initialized. Please prepare documents first."

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tipsy-topsy-dashboard.com",
                "X-Title": "Tipsy Topsy Analytics Dashboard"
            }

            summary_prompt = f"""Generate a comprehensive sales summary report using ONLY the business data provided below.
CRITICAL: Use ONLY the data shown - do not add, invent, or estimate any information not present in the context.

{self.context}

STRICT REQUIREMENTS:
- ONLY use numbers and data explicitly shown above
- For size information, reference the detailed transaction sections if available
- For return analysis, use both aggregated data and detailed transactions
- If specific data is not available in context, state "Data not available" instead of estimating
- Do not make assumptions about trends, causes, or future performance
- Base all insights strictly on the actual data provided

Please structure the summary with:
1. Executive Summary (based only on provided data)
2. Key Performance Metrics (only using provided numbers)
3. Sales Trends & Analysis (only using provided data)
4. Return Analysis & Insights (only using provided return data and sizes where available)
5. Brand Performance Overview (only using provided brand data)
6. Recommendations for Improvement (based only on provided data, no speculative recommendations)

Make it professional but stick STRICTLY to the data provided - no external knowledge or assumptions."""

            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a senior business analyst creating a comprehensive sales summary report. Use the provided data to generate insights and recommendations."},
                    {"role": "user", "content": summary_prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.2
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # Longer timeout for summary generation
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    summary = result['choices'][0]['message']['content']
                    return summary
                else:
                    return "Unable to generate summary - no response from API"
            else:
                return f"API Error {response.status_code}: {response.text}"

        except Exception as e:
            return f"Error generating sales summary: {str(e)}"

def create_dashboard(data_processor: DataProcessor, rag_system: RAGSystem):
    """Create the main dashboard"""

    st.markdown('<div class="main-header">👕 Tipsy Topsy Analytics Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 Dashboard Controls</div>', unsafe_allow_html=True)

        # File upload
        st.subheader("📁 Data Source")
        uploaded_file = st.file_uploader("Upload vzrm_6.csv file", type="csv")

        if uploaded_file is not None:
            if data_processor.load_and_process_data(uploaded_file):
                st.success("✅ Data loaded successfully!")

                # Date filtering
                st.subheader("📅 Date Filter")
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date",
                                             value=data_processor.df['Bill Date'].min().date() if hasattr(data_processor, 'df') and data_processor.df is not None else datetime.now().date())
                with col2:
                    end_date = st.date_input("End Date",
                                           value=data_processor.df['Bill Date'].max().date() if hasattr(data_processor, 'df') and data_processor.df is not None else datetime.now().date())

                if start_date and end_date:
                    # Date validation
                    if start_date > end_date:
                        st.error("❌ Invalid date range! Start date must be before or equal to end date.")
                        st.stop()

                    # Filter data by date range
                    mask = (data_processor.df['Bill Date'].dt.date >= start_date) & (data_processor.df['Bill Date'].dt.date <= end_date)
                    filtered_df = data_processor.df[mask].copy()

                    if filtered_df.empty:
                        st.warning(f"⚠️ No data found for the selected date range: {start_date} to {end_date}")
                        st.stop()

                    # Update sales and returns based on filtered data using transaction type
                    data_processor.sales_df = filtered_df[filtered_df['Tran Type'] == 'Sales'].copy()
                    data_processor.returns_df = filtered_df[filtered_df['Tran Type'] == 'Return'].copy()

                    # Clean return values for filtered data
                    if not data_processor.returns_df.empty:
                        data_processor.returns_df['Value'] = data_processor.returns_df['Value'].astype(str).str.replace(r'[()]', '', regex=True).astype(float)
                        data_processor.returns_df['Qty'] = data_processor.returns_df['Qty'].abs()

                    # Regenerate insights with filtered data
                    data_processor._generate_insights()

                    st.success(f"📊 Data filtered: {len(filtered_df)} transactions from {start_date} to {end_date}")

                # RAG System
                if RAG_AVAILABLE:
                    st.subheader("🤖 AI Assistant")
                    api_key = st.text_input("OpenRouter API Key", type="password", help="Get your API key from https://openrouter.ai/", value="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")
                    if api_key:
                        rag_system = RAGSystem(api_key)
                        context = rag_system.prepare_documents(data_processor)
                        st.success("🤖 AI Assistant ready!")

        # Quick stats
        if hasattr(data_processor, 'insights_cache') and data_processor.insights_cache:
            insights = data_processor.insights_cache
            st.subheader("📈 Quick Stats")
            st.metric("Total Sales", f"{insights['total_sales']:,}")
            st.metric("Net Revenue", f"₹{insights['net_revenue']:,.0f}")
            return_count = insights.get('return_count', 0)
            st.metric("Return Rate", f"{insights['return_rate']:.2f}% ({format_indian_number(return_count)} returns)")
            st.metric("Unique Products", f"{insights['unique_products']}")

    # Main content
    if not hasattr(data_processor, 'df') or data_processor.df is None:
        st.info("👆 Please upload your vzrm_6.csv file to begin analysis.")
        return

    # Key Metrics Row
    insights = data_processor.get_insights()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_sales_qty = data_processor.sales_df['Qty'].sum() if data_processor.sales_df is not None else 0
        total_return_qty = abs(data_processor.returns_df['Qty'].sum()) if data_processor.returns_df is not None else 0
        net_qty = total_sales_qty - total_return_qty
        st.metric("Total Net Items Sold", f"{net_qty:,}", f"Sales: {total_sales_qty:,}")

    with col2:
        total_sales_value = data_processor.sales_df['Value'].sum() if data_processor.sales_df is not None else 0
        total_return_value = abs(data_processor.returns_df['Value'].sum()) if data_processor.returns_df is not None else 0
        net_value = total_sales_value - total_return_value
        st.metric("Total Net Revenue", f"₹{net_value:,.0f}", f"Sales: ₹{total_sales_value:,.0f}")

    with col3:
        unique_products = len(data_processor.sales_df['Product Code'].unique()) if data_processor.sales_df is not None else 0
        st.metric("Unique Products Sold", f"{unique_products}")

    with col4:
        # Calculate return rate based on the actual data
        total_sales_qty = data_processor.sales_df['Qty'].sum() if data_processor.sales_df is not None else 0
        total_return_qty = abs(data_processor.returns_df['Qty'].sum()) if data_processor.returns_df is not None else 0
        actual_return_rate = (total_return_qty / total_sales_qty * 100) if total_sales_qty > 0 else 0
        return_count = int(total_return_qty)
        st.metric("Overall Return Rate", f"{actual_return_rate:.2f}% ({format_indian_number(return_count)} returns)")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏆 Performance Analysis",
        "📊 Brand Analysis",
        "📈 Product Analysis",
        "🔄 Returns Analysis",
        "🤖 AI Insights",
        "💡 Business Insights",
        "📋 Data Overview"
    ])

    with tab1:
        create_performance_analysis(data_processor)

    with tab2:
        create_brand_analysis(data_processor)

    with tab3:
        create_product_analysis(data_processor)

    with tab4:
        create_returns_analysis(data_processor)

    with tab5:
        create_ai_insights(data_processor, rag_system)

    with tab6:
        create_business_insights(data_processor)

    with tab7:
        create_data_overview(data_processor)

def create_performance_analysis(data_processor: DataProcessor):
    """Create performance analysis tab"""
    st.markdown('<div class="section-header">🏆 Top Performers</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Fastest Selling Products (Qty/Day)")
        # Calculate daily rates using sales data
        sales_df = data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame()
        if not sales_df.empty:
            min_date = sales_df['Bill Date'].min()
            max_date = sales_df['Bill Date'].max()
            days_in_period = (max_date - min_date).days + 1

            daily_rates = sales_df.groupby(['Brand Code', 'Product Code']).agg({
                'Qty': 'sum'
            }).reset_index()

            daily_rates['Qty_Per_Day'] = (daily_rates['Qty'] / days_in_period).round(2)
            daily_rates['Brand_Product'] = daily_rates['Brand Code'] + ' - ' + daily_rates['Product Code']
            daily_rates = daily_rates.sort_values('Qty_Per_Day', ascending=False)

            if not daily_rates.empty:
                top_fastest = daily_rates.head(10)

                fig_fastest = px.bar(
                    top_fastest,
                    x='Qty_Per_Day',
                    y='Brand_Product',
                    title="Top 10 Fastest Selling Products",
                    color='Qty_Per_Day',
                    color_continuous_scale='viridis',
                    orientation='h'
                )
                fig_fastest.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fastest, width='stretch')

                # Display table
                st.dataframe(top_fastest[['Brand Code', 'Product Code', 'Qty', 'Qty_Per_Day']].head(5))

    with col2:
        st.subheader("🥇 Best Selling Products (Total Qty)")
        if not data_processor.df.empty:
            # Calculate net data similar to new.py
            sales_only = data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame()
            returns_only = data_processor.returns_df if data_processor.returns_df is not None else pd.DataFrame()

            if not sales_only.empty:
                # Group sales by product
                sales_summary = sales_only.groupby(['Brand Code', 'Product Code']).agg({
                    'Qty': 'sum',
                    'Value': 'sum'
                }).reset_index()

                # Group returns by product
                if not returns_only.empty:
                    returns_summary = returns_only.groupby(['Brand Code', 'Product Code']).agg({
                        'Qty': lambda x: abs(x.sum()),
                        'Value': lambda x: abs(x.sum())
                    }).reset_index()
                    returns_summary.columns = ['Brand Code', 'Product Code', 'Return_Qty', 'Return_Value']

                    # Merge sales and returns
                    net_data = pd.merge(sales_summary, returns_summary[['Brand Code', 'Product Code', 'Return_Qty', 'Return_Value']],
                                      on=['Brand Code', 'Product Code'], how='left')
                    net_data[['Return_Qty', 'Return_Value']] = net_data[['Return_Qty', 'Return_Value']].fillna(0)
                    net_data['Net_Qty'] = net_data['Qty'] - net_data['Return_Qty']
                    net_data['Net_Value'] = net_data['Value'] - net_data['Return_Value']
                    net_data['Return_Rate'] = (net_data['Return_Qty'] / net_data['Qty'] * 100).round(2)
                else:
                    net_data = sales_summary.copy()
                    net_data['Return_Qty'] = 0
                    net_data['Return_Value'] = 0
                    net_data['Net_Qty'] = net_data['Qty']
                    net_data['Net_Value'] = net_data['Value']
                    net_data['Return_Rate'] = 0

                if not net_data.empty:
                    best_selling = net_data.nlargest(10, 'Net_Qty')
                    best_selling['Brand_Product'] = best_selling['Brand Code'] + ' - ' + best_selling['Product Code']

                    fig_best = px.bar(
                        best_selling,
                        x='Net_Qty',
                        y='Brand_Product',
                        title="Top 10 Best Selling Products",
                        color='Net_Qty',
                        color_continuous_scale='plasma',
                        orientation='h'
                    )
                    fig_best.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_best, width='stretch')

                    # Display table
                    st.dataframe(best_selling[['Brand Code', 'Product Code', 'Net_Qty', 'Return_Rate']].head(5))

    # Bottom performers section
    if not sales_only.empty:
        st.subheader("⚠️ Bottom 5 Products (Need Attention)")
        if not net_data.empty:
            bottom_products = net_data.nsmallest(5, 'Net_Qty')
            bottom_products['Brand_Product'] = bottom_products['Brand Code'] + ' - ' + bottom_products['Product Code']

            fig_bottom = px.bar(
                bottom_products,
                x='Net_Qty',
                y='Brand_Product',
                title="Bottom 5 Products by Net Quantity",
                color='Net_Qty',
                color_continuous_scale='reds',
                orientation='h'
            )
            fig_bottom.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bottom, width='stretch')

def create_brand_analysis(data_processor: DataProcessor):
    """Create brand analysis tab"""
    st.markdown('<div class="section-header">📊 Brand Analysis</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Brand Performance by Quantity")
        # Calculate brand analysis similar to new.py
        sales_df = data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame()
        returns_df = data_processor.returns_df if data_processor.returns_df is not None else pd.DataFrame()

        if not sales_df.empty:
            # Sales by brand
            brand_sales = sales_df.groupby('Brand Code').agg({
                'Qty': 'sum',
                'Value': 'sum',
                'Bill No': 'nunique'
            }).reset_index()

            # Returns by brand
            if not returns_df.empty:
                brand_returns = returns_df.groupby('Brand Code').agg({
                    'Qty': lambda x: abs(x.sum()),
                    'Value': lambda x: abs(x.sum())
                }).reset_index()
                brand_returns.columns = ['Brand Code', 'Return_Qty', 'Return_Value']

                # Merge brand data
                brand_analysis = pd.merge(brand_sales, brand_returns, on='Brand Code', how='left')
                brand_analysis[['Return_Qty', 'Return_Value']] = brand_analysis[['Return_Qty', 'Return_Value']].fillna(0)

                # Calculate metrics
                brand_analysis['Net_Qty'] = brand_analysis['Qty'] - brand_analysis['Return_Qty']
                brand_analysis['Net_Value'] = brand_analysis['Value'] - brand_analysis['Return_Value']
                brand_analysis['Return_Rate'] = (brand_analysis['Return_Qty'] / brand_analysis['Qty'] * 100).round(2)

                # Calculate consistency (coefficient of variation) - lower is better
                brand_daily_sales = sales_df.groupby(['Brand Code', 'Bill Date']).agg({
                    'Qty': 'sum'
                }).reset_index()

                # Calculate CV for each brand
                consistency_metrics = []
                for brand in brand_analysis['Brand Code'].unique():
                    brand_data = brand_daily_sales[brand_daily_sales['Brand Code'] == brand]
                    if len(brand_data) > 1:
                        mean_qty = brand_data['Qty'].mean()
                        std_qty = brand_data['Qty'].std()
                        cv = (std_qty / mean_qty * 100) if mean_qty > 0 else 0
                    else:
                        cv = 0
                    consistency_metrics.append({
                        'Brand Code': brand,
                        'CV': round(cv, 2),
                        'Sales_Days': len(brand_data)
                    })

                consistency_df = pd.DataFrame(consistency_metrics)

                # Merge with brand analysis
                brand_analysis = pd.merge(brand_analysis, consistency_df, on='Brand Code', how='left')
                brand_analysis['CV'] = brand_analysis['CV'].fillna(0)

                # Calculate reliability score (0-100, higher is better)
                brand_analysis['CV_normalized'] = np.minimum(brand_analysis['CV'], 100)
                brand_analysis['Reliability_Score'] = (
                    (100 - brand_analysis['Return_Rate']) * (100 - brand_analysis['CV_normalized']) / 100
                ).round(2)
            else:
                brand_analysis = brand_sales.copy()
                brand_analysis['Return_Qty'] = 0
                brand_analysis['Return_Value'] = 0
                brand_analysis['Net_Qty'] = brand_analysis['Qty']
                brand_analysis['Net_Value'] = brand_analysis['Value']
                brand_analysis['Return_Rate'] = 0
                brand_analysis['CV'] = 0
                brand_analysis['Sales_Days'] = 0
                brand_analysis['Reliability_Score'] = 100.0

            if not brand_analysis.empty:
                fig_brand_qty = px.bar(
                    brand_analysis.sort_values('Net_Qty', ascending=False),
                    x='Brand Code',
                    y='Net_Qty',
                    title="Net Quantity Sold by Brand",
                    color='Net_Qty',
                    color_continuous_scale='blues'
                )
                fig_brand_qty.update_layout(height=400)
                st.plotly_chart(fig_brand_qty, width='stretch')

    with col2:
        st.subheader("💰 Brand Performance by Revenue")
        if not brand_analysis.empty:
            fig_brand_rev = px.bar(
                brand_analysis.sort_values('Net_Value', ascending=False),
                x='Brand Code',
                y='Net_Value',
                title="Net Revenue by Brand",
                color='Net_Value',
                color_continuous_scale='greens'
            )
            fig_brand_rev.update_layout(height=400)
            st.plotly_chart(fig_brand_rev, width='stretch')

    # Brand reliability analysis
    st.subheader("🔒 Brand Reliability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if not brand_analysis.empty:
            fig_reliability = px.scatter(
                brand_analysis,
                x='Return_Rate',
                y='CV',
                size='Net_Qty',
                hover_data=['Brand Code', 'Reliability_Score'],
                title="Brand Reliability: Return Rate vs Sales Consistency",
                color='Reliability_Score',
                color_continuous_scale='RdYlGn'
            )
            fig_reliability.update_layout(height=400)
            st.plotly_chart(fig_reliability, width='stretch')

    with col2:
        if not brand_analysis.empty:
            reliable_brands = brand_analysis.sort_values('Reliability_Score', ascending=False)
            st.subheader("🌟 Most Reliable Brands")
            st.dataframe(reliable_brands[['Brand Code', 'Net_Qty', 'Return_Rate', 'CV', 'Reliability_Score']].head(10))

def create_product_analysis(data_processor: DataProcessor):
    """Create product analysis tab"""
    st.markdown('<div class="section-header">📈 Product Analysis</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥇 Top Products by Quantity")
        if insights['top_products_qty']:
            products_df = pd.DataFrame(insights['top_products_qty'])
            products_df['Brand_Product'] = products_df['Brand Code'] + ' - ' + products_df['Product Code']
            fig = px.bar(products_df, x='Qty', y='Brand_Product', orientation='h', title="Top Products by Quantity")
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("💎 Top Products by Revenue")
        if insights['top_products_value']:
            products_value_df = pd.DataFrame(insights['top_products_value'])
            products_value_df['Brand_Product'] = products_value_df['Brand Code'] + ' - ' + products_value_df['Product Code']
            fig = px.bar(products_value_df, x='Value', y='Brand_Product', orientation='h', title="Top Products by Revenue")
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')

def create_returns_analysis(data_processor: DataProcessor):
    """Create returns analysis tab"""
    st.markdown('<div class="section-header">🔄 Returns Analysis</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    if not insights['return_analysis']:
        st.info("No returns data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Products with Most Returns")
        returns_df = pd.DataFrame(insights['return_analysis'])
        returns_df['Brand_Product'] = returns_df['Brand Code'] + ' - ' + returns_df['Product Code']
        top_returns = returns_df.nlargest(10, 'Qty')
        fig = px.bar(top_returns, x='Qty', y='Brand_Product', orientation='h', title="Products with Most Returns")
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("💸 Return Value Analysis")
        top_returns_value = returns_df.nlargest(10, 'Value')
        fig = px.bar(top_returns_value, x='Value', y='Brand_Product', orientation='h', title="Products with Highest Return Value")
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')

def create_ai_insights(data_processor: DataProcessor, rag_system: RAGSystem):
    """Create AI insights tab with chatbot"""
    st.markdown('<div class="section-header">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)

    if not RAG_AVAILABLE:
        st.warning("AI features require additional packages. Please install requirements.txt")
        return

    # Show Current RAG Context
    st.subheader("📋 Current RAG Context (Data Available to AI)")

    if rag_system and rag_system.is_initialized:
        with st.expander("🔍 View Context Data Passed to AI", expanded=False):
            st.code(rag_system.context, language="text")
            st.info("💡 This is exactly what the AI model sees and can use to answer questions. It contains NO external knowledge.")

    # Chat interface
    st.subheader("💬 Ask Questions About Your Data")

    if rag_system and rag_system.is_initialized:
        # Chat messages container
        chat_container = st.container()

        # Chat input
        user_input = st.chat_input("Ask me anything about your sales data...")

        if user_input:
            # Get AI response
            response = rag_system.query(user_input)

            # Store in session state for persistence
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []

            st.session_state.chat_messages.append({"user": user_input, "bot": response})

        # Display chat messages
        with chat_container:
            if 'chat_messages' in st.session_state:
                for message in st.session_state.chat_messages:
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message['user']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>AI Assistant:</strong> {message['bot']}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("🤖 Please enter your OpenRouter API key in the sidebar to enable AI assistant.")

    # Sales Summary Generation
    st.subheader("📊 AI-Generated Sales Summary")

    if rag_system and rag_system.is_initialized:
        if st.button("Generate Comprehensive Sales Summary", type="primary"):
            with st.spinner("Generating sales summary using GPT-4o..."):
                summary = rag_system.generate_sales_summary()

                if summary and not summary.startswith("Error"):
                    st.success("✅ Sales summary generated successfully!")

                    # Display the summary in a nice format
                    st.markdown("### 📈 Executive Sales Summary")
                    st.markdown(summary)

                    # Add download option
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="sales_summary_report.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"❌ Failed to generate summary: {summary}")
    else:
        st.info("Please initialize the AI assistant first by entering your API key.")

    # Predefined insights
    st.subheader("🔍 Quick Insights")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generate Sales Summary"):
            insights = data_processor.get_insights()
            summary = f"""
            **Sales Summary:**
            - Total transactions: {insights['total_sales']:,}
            - Total revenue: ₹{insights['total_revenue']:,.0f}
            - Date range: {insights['date_range']['start']} to {insights['date_range']['end']}
            - Unique products: {insights['unique_products']}
            - Top brand by quantity: {insights['top_brands_qty'][0]['Brand Code'] if insights['top_brands_qty'] else 'N/A'}
            """
            st.markdown(summary)

    with col2:
        if st.button("🔄 Generate Returns Analysis"):
            insights = data_processor.get_insights()
            if insights['return_analysis']:
                returns_summary = f"""
                **Returns Analysis:**
                - Total returns: {insights['total_returns']:,}
                - Return rate: {insights['return_rate']:.2f}% ({format_indian_number(insights.get('return_count', 0))} returns)
                - Total return value: ₹{insights['total_return_value']:,.0f}
                - Most returned product: {insights['return_analysis'][0]['Brand Code']} - {insights['return_analysis'][0]['Product Code']} ({insights['return_analysis'][0]['Qty']} returns)
                """
                st.markdown(returns_summary)
            else:
                st.info("No returns data available.")

def create_business_insights(data_processor: DataProcessor):
    """Create business insights tab"""
    st.markdown('<div class="section-header">💡 Business Insights</div>', unsafe_allow_html=True)

    if not hasattr(data_processor, 'insights_cache') or not data_processor.insights_cache:
        st.info("📊 Analyzing your data to generate insights...")
        return

    # Initialize insights generator
    insights_gen = InsightsGenerator(data_processor)
    business_insights = insights_gen.generate_all_insights()

    # Display formatted insights
    st.markdown(insights_gen.get_formatted_insights())

    # Detailed insights sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Performance Analysis")
        perf_insights = business_insights['performance']

        st.write(f"**Summary:** {perf_insights['summary']}")
        st.write(f"**Assessment:** {perf_insights['return_assessment']}")

        # Key metrics in a nice format
        st.write("**Key Metrics:**")
        metrics_df = pd.DataFrame([
            ["Total Transactions", f"{perf_insights['key_metrics']['total_transactions']:,}"],
            ["Total Revenue", f"₹{perf_insights['key_metrics']['total_revenue']:,.0f}"],
            ["Net Revenue", f"₹{perf_insights['key_metrics']['net_revenue']:,.0f}"],
            ["Return Rate", f"{perf_insights['key_metrics']['return_rate']:.2f}% ({format_indian_number(data_processor.get_insights().get('return_count', 0))} returns)"]
        ], columns=["Metric", "Value"])
        st.dataframe(metrics_df, width='stretch')

    with col2:
        st.subheader("🎯 Top Recommendations")
        recommendations = business_insights['recommendations']

        for i, rec in enumerate(recommendations[:5], 1):
            with st.expander(f"{i}. {rec['priority']} - {rec['title']}"):
                st.write(f"**Description:** {rec['description']}")
                st.write("**Action Items:**")
                for action in rec['actions']:
                    st.write(f"• {action}")

    # Product and brand insights
    st.subheader("🏷️ Product & Brand Insights")

    col1, col2 = st.columns(2)

    with col1:
        product_insights = business_insights['products']
        if product_insights['top_performers']:
            st.write("**Top Performing Products:**")
            for performer in product_insights['top_performers'][:3]:
                st.write(f"• {performer['product']}: {performer['quantity']:,} units - {performer['impact']}")

        brand_insights = business_insights['brands']
        if brand_insights['top_brands']:
            st.write("**Top Brands:**")
            for brand in brand_insights['top_brands'][:3]:
                st.write(f"• {brand['brand']}: {brand['quantity']:,} units - {brand['recommendation']}")

    with col2:
        returns_insights = business_insights['returns']
        if returns_insights['return_analysis']:
            st.write("**Return Analysis:**")
            st.write(f"• Total Returns: {returns_insights['return_analysis']['total_returns']:,}")
            return_rate = data_processor.get_insights()['return_rate']
            return_count = data_processor.get_insights().get('return_count', 0)
            st.write(f"• Return Rate: {return_rate:.2f}% ({format_indian_number(return_count)} returns)")
            st.write(f"• Average Return Value: ₹{returns_insights['return_analysis']['avg_return_value']:,.0f}")

        trend_insights = business_insights['trends']
        if trend_insights['sales_trends']:
            st.write("**Sales Trend:**")
            trend = trend_insights['sales_trends']
            st.write(f"• Direction: {trend['direction']}")
            st.write(f"• Change: {trend['change_percent']:.1f}%")

    # Customer insights
    st.subheader("👥 Customer Insights")

    customer_insights = business_insights['customer']

    col1, col2 = st.columns(2)

    with col1:
        if customer_insights['size_preferences']:
            st.write("**Size Preferences:**")
            st.write(f"Most Popular Sizes: {', '.join(customer_insights['size_preferences']['most_popular_sizes'])}")
            st.write(f"Recommendation: {customer_insights['size_preferences']['recommendation']}")

    with col2:
        if customer_insights['purchase_patterns']:
            patterns = customer_insights['purchase_patterns']
            st.write("**Purchase Patterns:**")
            st.write(f"• Repeat Customer Rate: {patterns['repeat_customer_rate']:.1f}%")
            st.write(f"• Average Purchase Value: ₹{patterns['avg_purchase_value']:,.0f}")

            if patterns['repeat_customer_rate'] < 20:
                st.warning("Low repeat customer rate - consider implementing loyalty programs")

    # Action plan
    st.subheader("📋 Action Plan")
    st.write("Based on the analysis, here are the **top 3 priorities** for your business:")

    priorities = []
    for rec in recommendations[:3]:
        priorities.append(f"**{rec['priority']} Priority:** {rec['title']} - {rec['description']}")

    for priority in priorities:
        st.write(f"• {priority}")

    # Export insights
    st.subheader("💾 Export Insights")

    insights_text = insights_gen.get_formatted_insights()

    st.download_button(
        label="📄 Download Business Insights",
        data=insights_text,
        file_name="business_insights.txt",
        mime="text/plain"
    )

def create_data_overview(data_processor: DataProcessor):
    """Create data overview tab"""
    st.markdown('<div class="section-header">📋 Data Overview</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    # Basic information
    st.subheader("📊 Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Data Statistics:**")
        st.write(f"- Total Records: {len(data_processor.df):,}")
        st.write(f"- Date Range: {insights['date_range']['start']} to {insights['date_range']['end']}")
        st.write(f"- Sales Records: {insights['total_sales']:,}")
        st.write(f"- Returns Records: {insights['total_returns']:,}")

    with col2:
        st.write("**Business Metrics:**")
        st.write(f"- Total Revenue: ₹{insights['total_revenue']:,.0f}")
        st.write(f"- Net Revenue: ₹{insights['net_revenue']:,.0f}")
        st.write(f"- Return Rate: {insights['return_rate']:.2f}% ({format_indian_number(insights.get('return_count', 0))} returns)")
        st.write(f"- Unique Products: {insights['unique_products']}")
        st.write(f"- Unique Brands: {insights['unique_brands']}")

    # Sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(data_processor.df.head(20))

    # Download options
    st.subheader("💾 Export Data")

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = convert_df(data_processor.df)
        st.download_button(
            label="📄 Download Full Dataset",
            data=csv_data,
            file_name='tipsy_topsy_full_data.csv',
            mime='text/csv',
        )

    with col2:
        if data_processor.sales_df is not None:
            csv_sales = convert_df(data_processor.sales_df)
            st.download_button(
                label="📈 Download Sales Data",
                data=csv_sales,
                file_name='tipsy_topsy_sales_data.csv',
                mime='text/csv',
            )

    with col3:
        if data_processor.returns_df is not None:
            csv_returns = convert_df(data_processor.returns_df)
            st.download_button(
                label="🔄 Download Returns Data",
                data=csv_returns,
                file_name='tipsy_topsy_returns_data.csv',
                mime='text/csv',
            )

def main():
    """Main function"""
    # Initialize components
    data_processor = DataProcessor()
    rag_system = None

    create_dashboard(data_processor, rag_system)

if __name__ == "__main__":
    main()
