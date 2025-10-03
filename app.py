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

# Brand code to brand name mapping
BRAND_NAME_MAPPING = {
    'CELIO': 'Celio',
    'SPYKAR': 'Spykar',
    'IT MENS': 'IT Mens',
    'KAAPUS': 'Kaapus',
    'BH': 'BH',
    'JACK&JONES': 'Jack & Jones',
    'NUMEROUNO MENS': 'Numero Uno Mens',
    'FLYING MACHINE': 'Flying Machine',
    'US POLO': 'US Polo',
    'MUFTI': 'Mufti',
    'ARROW N': 'Arrow New York',
    'ARROW': 'Arrow',
    'ARROW SPORTS': 'Arrow Sports',
    'ROOKIES': 'Rookies',
    'BENETTON': 'Benetton',
    'VOI JEANS': 'VOI Jeans',
    'LINEN CLUB': 'Linen Club',
    'JOCKEY': 'Jockey',
    'BLACK BERRY': 'Black Berry',
    'LEVIS': 'Levis',
    'RARE RABBIT': 'Rare Rabbit',
    'CFS': 'CFS',
    'DENVER': 'Denver',
    'JAGUAR': 'Jaguar',
    'FAHRENHEIT': 'Fahrenheit',
    'VH': 'VH',
    'LC': 'LC',
    'BRUT': 'Brut',
    'INDIAN TERRAIN': 'Indian Terrain',
    'LEBUCK': 'Lebuck'
}

def get_brand_name(brand_code):
    """Get brand name from brand code"""
    if pd.isna(brand_code):
        return 'Unknown'
    brand_code_str = str(brand_code).strip()
    # If it looks like a product code (starts with numbers), use generic name
    if brand_code_str.replace('P', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '') == '':
        return 'Generic Brand'
    return BRAND_NAME_MAPPING.get(brand_code_str, brand_code_str)

def get_product_categories(df):
    """Get available product categories from dataframe"""
    if df is None or df.empty:
        return {}

    category_keywords = {
        'T-Shirts': ['T-SHIRT', 'TEE', 'T SHIRT'],
        'Shirts': ['SHIRT'],
        'Trousers': ['TROUSER', 'PANT'],
        'Jeans': ['JEANS', 'DENIM'],
        'Jackets': ['JACKET', 'BLAZER'],
        'Others': []
    }

    categories_found = {}
    for category, keywords in category_keywords.items():
        if keywords:
            # Check if any products match this category
            mask = df['Product Code'].str.contains('|'.join(keywords), case=False, na=False)
            if mask.any():
                categories_found[category] = True

    return categories_found

def get_filtered_data(df, brand=None, category=None, color=None, size=None):
    """Apply multi-level filtering to dataframe"""
    filtered_df = df.copy()

    # Brand filter
    if brand and brand != "All Brands":
        filtered_df = filtered_df[filtered_df['Brand Code'] == brand]

    # Category filter
    if category and category != "All Categories":
        category_keywords = {
            'T-Shirts': ['T-SHIRT', 'TEE', 'T SHIRT'],
            'Shirts': ['SHIRT'],
            'Trousers': ['TROUSER', 'PANT'],
            'Jeans': ['JEANS', 'DENIM'],
            'Jackets': ['JACKET', 'BLAZER']
        }

        if category in category_keywords:
            keywords = category_keywords[category]
            mask = filtered_df['Product Code'].str.contains('|'.join(keywords), case=False, na=False)
            filtered_df = filtered_df[mask]

    # Color filter
    if color and color != "All Colors" and 'Shade Code' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Shade Code'] == color]

    # Size filter
    if size and size != "All Sizes" and 'Size' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Size'] == size]

    return filtered_df

class DataProcessor:
    """Unified data processor for vzrm_6.csv file"""

    def __init__(self):
        self.df = None
        self.sales_df = None
        self.returns_df = None
        self.processed_data = {}
        self.insights_cache = {}

    def load_and_process_data(self, uploaded_file) -> bool:
        """Load and process data file (CSV, XLSX, or XLSB)"""
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                # Read CSV file with better error handling
                try:
                    # Try UTF-8 first
                    self.df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        self.df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
                    except Exception as e:
                        st.error(f"❌ Error reading CSV file: {str(e)}")
                        st.info("💡 Try saving your CSV file with UTF-8 encoding, or use Excel format.")
                        return False
            elif file_extension in ['xlsx', 'xlsb']:
                # Read Excel file with better error handling for large files
                try:
                    # Try reading as XLSX first
                    if file_extension == 'xlsx':
                        try:
                            # Use read_excel with better options for large files
                            self.df = pd.read_excel(
                                uploaded_file,
                                engine='openpyxl',
                                dtype=str,  # Read everything as string first to preserve data
                                keep_default_na=False,  # Don't convert empty cells to NaN
                                na_filter=False  # Don't filter NaN values
                            )
                            st.info(f"✅ Successfully loaded {len(self.df):,} rows from Excel file")
                        except Exception as e:
                            st.warning(f"⚠️ Primary Excel reading failed ({str(e)}), trying alternative method...")
                            # Fallback to basic reading
                            xl = pd.ExcelFile(uploaded_file, engine='openpyxl')
                            if len(xl.sheet_names) > 1:
                                st.warning(f"⚠️ Excel file contains multiple sheets: {xl.sheet_names}")
                                st.info("💡 Using the first sheet. If you need a different sheet, please save it as a separate file.")
                            self.df = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name=0)
                    else:  # xlsb
                        # For XLSB files
                        try:
                            self.df = pd.read_excel(
                                uploaded_file,
                                engine='pyxlsb',
                                dtype=str,  # Read everything as string first
                                keep_default_na=False,
                                na_filter=False
                            )
                            st.info(f"✅ Successfully loaded {len(self.df):,} rows from Excel binary file")
                        except Exception as e:
                            st.warning(f"⚠️ Primary XLSB reading failed ({str(e)}), trying alternative method...")
                            xl = pd.ExcelFile(uploaded_file, engine='pyxlsb')
                            if len(xl.sheet_names) > 1:
                                st.warning(f"⚠️ Excel file contains multiple sheets: {xl.sheet_names}")
                                st.info("💡 Using the first sheet. If you need a different sheet, please save it as a separate file.")
                            self.df = pd.read_excel(uploaded_file, engine='pyxlsb', sheet_name=0)
                except ImportError:
                    st.error("❌ Missing required packages for Excel files. Please install: pip install openpyxl pyxlsb")
                    return False
                except Exception as e:
                    st.error(f"❌ Error reading Excel file: {str(e)}")
                    st.info("💡 Try saving your Excel file as CSV format, or ensure the first sheet contains your data.")
                    return False
            else:
                st.error(f"❌ Unsupported file type: {file_extension}. Please upload CSV, XLSX, or XLSB files.")
                return False

            # Clean column names
            self.df.columns = self.df.columns.str.strip()

            # Store initial row count for logging
            initial_rows = len(self.df)

            # Ensure consistent column naming (handle case differences)
            column_mapping = {
                'bill date': 'Bill Date',
                'tran type': 'Tran Type',
                'bill prefix': 'Bill Prefix',
                'bill no': 'Bill No',
                'customer code': 'Customer Code',
                'item promo code': 'Item Promo Code',
                'bill promo code': 'Bill Promo Code',
                'stockno': 'StockNo',
                'product code': 'Product Code',
                'brand code': 'Brand Code',
                'style code': 'Style Code',
                'shade code': 'Shade Code',
                'size': 'Size',
                'hsn code': 'HSN Code',
                'cost': 'Cost',
                'mrp': 'MRP',
                'doc rate': 'Doc Rate',
                'qty': 'Qty',
                'value': 'Value',
                'addons': 'Addons',
                'deduction': 'Deduction',
                'tax perc.': 'Tax Perc.',
                'tax': 'Tax',
                'item - discount': 'Item - Discount',
                'bill - discount': 'Bill - Discount',
                'total - discount': 'Total - Discount',
                'old bill ref.': 'Old Bill Ref.'
            }

            # Rename columns to match expected format (strip spaces first)
            self.df.columns = [column_mapping.get(col.lower().strip(), col) for col in self.df.columns]

            # Convert date column - try each date individually with all formats to maximize parsing
            def parse_dates_maximally(date_series):
                """Parse dates by trying each date with all formats until one works"""
                parsed_dates = []
                date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y']

                for date_val in date_series:
                    if pd.isna(date_val) or str(date_val).strip() == '':
                        parsed_dates.append(pd.NaT)
                        continue

                    date_str = str(date_val).strip()
                    parsed = False

                    # Try each format
                    for fmt in date_formats:
                        try:
                            parsed_date = pd.to_datetime(date_str, format=fmt)
                            parsed_dates.append(parsed_date)
                            parsed = True
                            break
                        except:
                            continue

                    # If no format worked, try automatic parsing
                    if not parsed:
                        try:
                            parsed_date = pd.to_datetime(date_str)
                            parsed_dates.append(parsed_date)
                        except:
                            parsed_dates.append(pd.NaT)

                return pd.Series(parsed_dates, index=date_series.index)

            # Apply the improved date parsing
            self.df['Bill Date'] = parse_dates_maximally(self.df['Bill Date'])

            # Debug: Log after date processing
            after_date_rows = len(self.df)
            if initial_rows != after_date_rows:
                st.info(f"📊 After date processing: {after_date_rows:,} rows (removed {initial_rows - after_date_rows:,} invalid date rows)")

            # Keep ALL rows - don't filter out rows with invalid dates
            # Only use valid dates for date-based analysis, but preserve all transaction data
            valid_dates_count = (~self.df['Bill Date'].isna()).sum()
            total_rows = len(self.df)

            if valid_dates_count == 0:
                st.warning("⚠️ No valid dates found in data. All rows retained but date-based analysis will be limited.")
            elif valid_dates_count < total_rows:
                # Show sample of invalid dates for debugging
                invalid_dates = self.df[self.df['Bill Date'].isna()]['Bill Date'].head(5)
                if not invalid_dates.empty:
                    pass  # Keep the logic but remove the message

            # Remove summary rows and obviously invalid data
            if 'Tran Type' in self.df.columns:
                # Remove only obvious summary rows, but be less aggressive
                summary_indicators = ['*Sub Total*', '*Sub Total* - *30/01/2025*', 'Total', 'Grand Total']
                before_filter = len(self.df)
                self.df = self.df[~self.df['Tran Type'].isin(summary_indicators)]
                after_filter = len(self.df)
                if before_filter != after_filter:
                    st.info(f"📊 After summary row removal: {after_filter:,} rows (removed {before_filter - after_filter:,} summary rows)")

                # Keep all product codes - don't filter out any rows based on product code
                # The user wants ALL transaction data preserved

                # Also keep all brand codes - don't filter out rows based on brand code either
                # Preserve all transaction data as requested by user

            # Convert numeric columns - handle the concatenated values properly
            numeric_columns = ['Cost', 'MRP', 'Doc Rate', 'Qty', 'Value', 'Tax', 'Item - Discount', 'Bill - Discount', 'Total - Discount']
            for col in numeric_columns:
                if col in self.df.columns:
                    # First convert to string to handle any weird formatting
                    self.df[col] = self.df[col].astype(str)
                    # Remove any non-numeric characters except decimal points and minus signs
                    self.df[col] = self.df[col].str.replace(r'[^\d.-]', '', regex=True)
                    # Convert to numeric, ensuring integers for quantities
                    if col == 'Qty':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
                    else:
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
        insights['total_revenue'] = int(self.sales_df['Value'].sum()) if self.sales_df is not None else 0
        insights['total_return_value'] = int(abs(self.returns_df['Value'].sum())) if self.returns_df is not None else 0
        insights['net_revenue'] = insights['total_revenue'] - insights['total_return_value']
        insights['unique_products'] = self.sales_df['Product Code'].nunique() if self.sales_df is not None else 0
        insights['unique_brands'] = self.sales_df['Brand Code'].nunique() if self.sales_df is not None else 0

        # Store full dataset metrics (before any filtering)
        full_sales_df = self.df[self.df['Tran Type'] == 'Sales'].copy() if self.df is not None else None
        full_returns_df = self.df[self.df['Tran Type'] == 'Return'].copy() if self.df is not None else None

        insights['full_total_sales'] = len(full_sales_df) if full_sales_df is not None else 0
        insights['full_total_returns'] = len(full_returns_df) if full_returns_df is not None else 0
        insights['full_total_revenue'] = int(full_sales_df['Value'].sum()) if full_sales_df is not None and not full_sales_df.empty else 0
        insights['full_total_return_value'] = int(abs(full_returns_df['Value'].sum())) if full_returns_df is not None and not full_returns_df.empty else 0
        insights['full_net_revenue'] = insights['full_total_revenue'] - insights['full_total_return_value']
        insights['full_unique_products'] = full_sales_df['Product Code'].nunique() if full_sales_df is not None and not full_sales_df.empty else 0
        insights['full_unique_brands'] = full_sales_df['Brand Code'].nunique() if full_sales_df is not None and not full_sales_df.empty else 0
        insights['full_total_quantity'] = len(full_sales_df) if full_sales_df is not None else 0  # Number of transactions, not sum of Qty

        # Date range - use only valid dates for date-based analysis
        valid_dates_df = self.df[~self.df['Bill Date'].isna()]

        if not valid_dates_df.empty:
            valid_count = len(valid_dates_df)
            total_count = len(self.df)
            insights['date_range'] = {
                'start': valid_dates_df['Bill Date'].min().strftime('%d/%m/%Y'),
                'end': valid_dates_df['Bill Date'].max().strftime('%d/%m/%Y')
            }
        else:
            # If no valid dates, use placeholder
            insights['date_range'] = {
                'start': 'Unknown',
                'end': 'Unknown'
            }
            st.warning("⚠️ No valid dates found for date range calculation")

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
            total_sales_qty = int(self.sales_df['Qty'].sum()) if self.sales_df is not None else 0
            total_return_qty = int(abs(self.returns_df['Qty'].sum())) if self.returns_df is not None else 0
            insights['return_rate'] = (total_return_qty / total_sales_qty * 100) if total_sales_qty > 0 else 0
            insights['return_count'] = total_return_qty  # Store the actual count
        else:
            insights['return_analysis'] = []
            insights['return_rate'] = 0
            insights['return_count'] = 0

        # Daily sales pattern - only use valid dates
        if self.sales_df is not None and not self.sales_df.empty:
            valid_date_sales = self.sales_df[~self.sales_df['Bill Date'].isna()]
            if not valid_date_sales.empty:
                daily_sales = valid_date_sales.groupby('Bill Date').agg({
                    'Qty': 'sum',
                    'Value': 'sum'
                }).reset_index()
                insights['daily_sales_trend'] = daily_sales.to_dict('records')
            else:
                insights['daily_sales_trend'] = []
                st.warning("⚠️ No valid dates found for daily sales trend analysis")
        else:
            insights['daily_sales_trend'] = []

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
        """Prepare comprehensive context from all dashboard data for RAG"""
        documents = []

        try:
            # 1. CURRENT FILTER STATE (Most Important)
            current_filters = []
            if 'filter_brand' in st.session_state and st.session_state.filter_brand != "All Brands":
                current_filters.append(f"Brand: {get_brand_name(st.session_state.filter_brand)}")
            if 'filter_category' in st.session_state and st.session_state.filter_category != "All Categories":
                current_filters.append(f"Category: {st.session_state.filter_category}")
            if 'filter_color' in st.session_state and st.session_state.filter_color != "All Colors":
                current_filters.append(f"Color: {st.session_state.filter_color}")
            if 'filter_size' in st.session_state and st.session_state.filter_size != "All Sizes":
                current_filters.append(f"Size: {st.session_state.filter_size}")

            if current_filters:
                documents.append(f"CURRENTLY APPLIED FILTERS: {', '.join(current_filters)}")
            else:
                documents.append("CURRENTLY APPLIED FILTERS: No filters applied (showing all data)")

            # 2. BASIC BUSINESS OVERVIEW
            insights = data_processor.get_insights()
            basic_stats = f"""
BUSINESS OVERVIEW:
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

            # 3. TOP PRODUCTS INFORMATION
            if insights['top_products_qty']:
                top_products_text = "TOP PRODUCTS BY QUANTITY SOLD:\n"
                for i, product in enumerate(insights['top_products_qty'][:15], 1):
                    top_products_text += f"{i}. {product['Brand Code']} - {product['Product Code']}: {product['Qty']} units\n"
                documents.append(top_products_text)

            if insights['top_products_value']:
                top_products_value_text = "TOP PRODUCTS BY REVENUE:\n"
                for i, product in enumerate(insights['top_products_value'][:15], 1):
                    top_products_value_text += f"{i}. {product['Brand Code']} - {product['Product Code']}: ₹{product['Value']:,.0f}\n"
                documents.append(top_products_value_text)

            # 4. BRAND ANALYSIS
            if insights['top_brands_qty']:
                brands_text = "TOP BRANDS BY QUANTITY:\n"
                for i, brand in enumerate(insights['top_brands_qty'][:10], 1):
                    brands_text += f"{i}. {brand['Brand Code']}: {brand['Qty']} units\n"
                documents.append(brands_text)

            # 5. RETURN ANALYSIS
            if insights['return_analysis']:
                returns_text = "RETURN ANALYSIS (Aggregated by Brand-Product):\n"
                for ret in insights['return_analysis'][:20]:
                    returns_text += f"{ret['Brand Code']} - {ret['Product Code']}: {ret['Qty']} returns, ₹{ret['Value']:,.0f} value\n"
                documents.append(returns_text)

            # 6. DETAILED TRANSACTIONS WITH SIZES
            if data_processor.returns_df is not None and not data_processor.returns_df.empty:
                returns_with_sizes = data_processor.returns_df[['Brand Code', 'Product Code', 'Size', 'Qty', 'Value']].head(30)
                if not returns_with_sizes.empty:
                    detailed_returns_text = "\nDETAILED RETURN TRANSACTIONS (with sizes):\n"
                    for _, ret in returns_with_sizes.iterrows():
                        size_info = f" (Size: {ret['Size']})" if pd.notna(ret['Size']) and ret['Size'] != '' else ""
                        detailed_returns_text += f"{ret['Brand Code']} - {ret['Product Code']}{size_info}: {abs(ret['Qty'])} return(s), ₹{abs(ret['Value']):,.0f} value\n"
                    documents.append(detailed_returns_text)

            # 7. SALES TRANSACTIONS WITH SIZES
            if data_processor.sales_df is not None and not data_processor.sales_df.empty:
                sales_sample = data_processor.sales_df[['Brand Code', 'Product Code', 'Size', 'Qty', 'Value']].dropna().head(25)
                if not sales_sample.empty:
                    sales_sizes_text = "\nSAMPLE SALES TRANSACTIONS (showing available sizes):\n"
                    for _, sale in sales_sample.iterrows():
                        if pd.notna(sale['Size']) and sale['Size'] != '':
                            sales_sizes_text += f"{sale['Brand Code']} - {sale['Product Code']} (Size: {sale['Size']}): {sale['Qty']} units, ₹{sale['Value']:,.0f}\n"
                    documents.append(sales_sizes_text)

            # 8. DAILY SALES TREND
            if insights['daily_sales_trend']:
                daily_text = "DAILY SALES TREND:\n"
                for day in insights['daily_sales_trend'][:10]:  # Last 10 days
                    daily_text += f"{day['Bill Date'].strftime('%d/%m/%Y')}: {day['Qty']} units, ₹{day['Value']:,.0f}\n"
                documents.append(daily_text)
            else:
                documents.append("DAILY SALES TREND: No valid date data available for trend analysis")

            # 9. SIZE DISTRIBUTION
            if 'Size' in data_processor.sales_df.columns if data_processor.sales_df is not None else False:
                size_dist = data_processor.sales_df.groupby('Size')['Qty'].sum().sort_values(ascending=False).head(10)
                if not size_dist.empty:
                    size_text = "SIZE DISTRIBUTION (Top 10):\n"
                    for size, qty in size_dist.items():
                        if pd.notna(size) and size != '':
                            size_text += f"Size {size}: {qty} units sold\n"
                    documents.append(size_text)

            # 10. COLOR ANALYSIS
            if 'Shade Code' in data_processor.sales_df.columns if data_processor.sales_df is not None else False:
                color_dist = data_processor.sales_df.groupby('Shade Code')['Qty'].sum().sort_values(ascending=False).head(10)
                if not color_dist.empty:
                    color_text = "COLOR DISTRIBUTION (Top 10):\n"
                    for color, qty in color_dist.items():
                        if pd.notna(color) and color != '':
                            color_text += f"Color {color}: {qty} units sold\n"
                    documents.append(color_text)

            # 11. BRAND-SIZE PREFERENCES
            if data_processor.sales_df is not None and not data_processor.sales_df.empty:
                brand_size = data_processor.sales_df.groupby(['Brand Code', 'Size'])['Qty'].sum().reset_index()
                brand_size = brand_size[brand_size['Size'].notna() & (brand_size['Size'] != '')]
                brand_size = brand_size.sort_values(['Brand Code', 'Qty'], ascending=[True, False])

                brand_size_text = "BRAND-SIZE PREFERENCES (Top sizes per brand):\n"
                current_brand = None
                for _, row in brand_size.head(20).iterrows():
                    if current_brand != row['Brand Code']:
                        if current_brand is not None:
                            brand_size_text += "\n"
                        current_brand = row['Brand Code']
                        brand_size_text += f"{current_brand}:\n"
                    brand_size_text += f"  Size {row['Size']}: {row['Qty']} units\n"
                documents.append(brand_size_text)

            # 12. PRICE & DISCOUNT ANALYSIS
            if data_processor.df is not None and not data_processor.df.empty:
                sales_data = data_processor.df[data_processor.df['Tran Type'] == 'Sales']

                # Price ranges
                price_ranges = pd.cut(sales_data['Value'], bins=[0, 500, 1000, 2000, 5000, float('inf')],
                                    labels=['₹0-500', '₹500-1000', '₹1000-2000', '₹2000-5000', '₹5000+'])
                price_dist = price_ranges.value_counts().sort_index()

                price_text = "PRICE RANGE DISTRIBUTION:\n"
                for price_range, count in price_dist.items():
                    price_text += f"{price_range}: {count} items sold\n"
                documents.append(price_text)

                # Discount analysis by brand
                discount_analysis = sales_data.groupby('Brand Code').agg({
                    'MRP': 'mean',
                    'Value': 'mean',
                    'Qty': 'sum'
                }).reset_index()

                discount_analysis['Discount_Percent'] = ((discount_analysis['MRP'] - discount_analysis['Value']) / discount_analysis['MRP'] * 100)

                top_discount_brands = discount_analysis.nlargest(10, 'Discount_Percent')
                discount_text = "TOP DISCOUNT BRANDS:\n"
                for _, brand in top_discount_brands.iterrows():
                    discount_text += f"{brand['Brand Code']}: {brand['Discount_Percent']:.1f}% discount, {brand['Qty']} units sold\n"
                documents.append(discount_text)

            # 13. BUSINESS INSIGHTS (if available)
            try:
                from insights_generator import InsightsGenerator
                insights_gen = InsightsGenerator(data_processor)
                business_insights = insights_gen.generate_all_insights()

                # Add key insights
                if 'performance' in business_insights:
                    perf = business_insights['performance']
                    insights_text = f"BUSINESS PERFORMANCE INSIGHTS:\n{perf['summary']}\n\nAssessment: {perf['return_assessment']}\n"
                    documents.append(insights_text)

                if 'customer' in business_insights:
                    customer = business_insights['customer']
                    if 'size_preferences' in customer:
                        size_prefs = customer['size_preferences']
                        size_text = f"SIZE PREFERENCES:\nMost Popular: {', '.join(size_prefs['most_popular_sizes'])}\nRecommendation: {size_prefs['recommendation']}\n"
                        documents.append(size_text)

            except Exception as e:
                # If insights generator fails, continue without it
                pass

            # 14. CURRENT FILTERED DATA SUMMARY
            if data_processor.sales_df is not None and not data_processor.sales_df.empty:
                filtered_summary = f"""
CURRENT FILTERED DATA SUMMARY:
- Filtered Sales Records: {len(data_processor.sales_df):,}
- Filtered Unique Products: {data_processor.sales_df['Product Code'].nunique() if data_processor.sales_df is not None else 0}
- Filtered Unique Brands: {data_processor.sales_df['Brand Code'].nunique() if data_processor.sales_df is not None else 0}
- Filtered Total Quantity: {data_processor.sales_df['Qty'].sum() if data_processor.sales_df is not None else 0:,}
- Filtered Total Value: ₹{data_processor.sales_df['Value'].sum() if data_processor.sales_df is not None else 0:,.0f}
- Date Parsing: Successfully parsed dates using multiple formats (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, MM/DD/YYYY)
"""
                documents.append(filtered_summary)

            self.context = "\n\n" + "="*80 + "\n\n".join(documents)
            self.is_initialized = True
            return self.context

        except Exception as e:
            error_msg = f"Error preparing comprehensive documents: {str(e)}"
            st.error(error_msg)
            self.context = error_msg
            return self.context

    def initialize_rag(self, context: str):
        """Initialize the RAG system with context"""
        self.context = context
        self.is_initialized = True

    def query(self, question: str) -> str:
        """Query the RAG system using OpenRouter API directly"""
        if not self.api_key or len(self.api_key.strip()) <= 10:
            return "❌ AI Assistant not available. Please enter a valid OpenRouter API key in the sidebar."

        if not self.is_initialized:
            return "❌ RAG system not initialized. Please ensure data is loaded and context is prepared."

        # Check if API key appears to be the demo key (starts with sk-or-v1-your)
        if self.api_key.startswith('sk-or-v1-your'):
            return "❌ Demo API key detected. Please use a real OpenRouter API key from your account."

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
                    return "❌ Unable to get response from API - no choices in response"
            elif response.status_code == 401:
                return "❌ API Authentication Failed: Invalid API key or insufficient credits. Please check your OpenRouter API key."
            elif response.status_code == 429:
                return "❌ API Rate Limit Exceeded: Too many requests. Please try again later."
            elif response.status_code == 400:
                return "❌ API Request Error: Invalid request format. Please check your question."
            else:
                return f"❌ API Error {response.status_code}: {response.text}"

        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

    def get_context(self) -> str:
        """Get the current context data"""
        return self.context

    def generate_sales_summary(self) -> str:
        """Generate a comprehensive sales summary using GPT-4o"""
        if not self.api_key or len(self.api_key.strip()) <= 10:
            return "❌ AI Assistant not available. Please enter a valid OpenRouter API key in the sidebar."

        if not self.is_initialized:
            return "❌ RAG system not initialized. Please ensure data is loaded and context is prepared."

        # Check if API key appears to be the demo key (starts with sk-or-v1-your)
        if self.api_key.startswith('sk-or-v1-your'):
            return "❌ Demo API key detected. Please use a real OpenRouter API key from your account."

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
                    return "❌ Unable to generate summary - no response from API"
            elif response.status_code == 401:
                return "❌ API Authentication Failed: Invalid API key or insufficient credits. Please check your OpenRouter API key."
            elif response.status_code == 429:
                return "❌ API Rate Limit Exceeded: Too many requests. Please try again later."
            elif response.status_code == 400:
                return "❌ API Request Error: Invalid request format."
            else:
                return f"❌ API Error {response.status_code}: {response.text}"

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
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=["csv", "xlsx", "xlsb"],
            help="Supported formats: CSV (.csv), Excel (.xlsx), Excel Binary (.xlsb). Make sure the first sheet contains your data."
        )

        if uploaded_file is not None:
            if data_processor.load_and_process_data(uploaded_file):

                # Get valid date range for defaults (after data is loaded and parsed)
                valid_dates = data_processor.df['Bill Date'].dropna() if hasattr(data_processor, 'df') and data_processor.df is not None else pd.Series()
                if not valid_dates.empty:
                    default_start = valid_dates.min().date()
                    default_end = valid_dates.max().date()
                else:
                    default_start = default_end = datetime.now().date()
                    st.warning("⚠️ No valid dates found in data!")

                # Date filtering
                st.subheader("📅 Date Filter")
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=default_start, key="start_date_input")
                with col2:
                    end_date = st.date_input("End Date", value=default_end, key="end_date_input")

                # Apply filtering if dates are not set to the full data range
                full_date_range = (start_date == default_start) and (end_date == default_end)

                if full_date_range:
                    pass  # No message needed
                else:
                    st.info(f"🔄 Date filtering active: {start_date} to {end_date}")

                if start_date and end_date and not full_date_range:
                    # Date validation
                    if start_date > end_date:
                        st.error("❌ Invalid date range! Start date must be before or equal to end date.")
                        st.stop()

                    # Apply date filtering
                    st.info("🔄 Applying date filter...")

                    # Simplified and more robust date filtering
                    filtered_rows = []
                    for idx, row in data_processor.df.iterrows():
                        bill_date = row['Bill Date']
                        if pd.isna(bill_date):
                            continue

                        # Convert to date for comparison
                        if hasattr(bill_date, 'date'):
                            bill_date_only = bill_date.date()
                        else:
                            bill_date_only = bill_date

                        if start_date <= bill_date_only <= end_date:
                            filtered_rows.append(idx)

                    filtered_df = data_processor.df.loc[filtered_rows].copy() if filtered_rows else data_processor.df.head(0).copy()

                    st.info(f"📅 Filtering data from {start_date} to {end_date} - found {len(filtered_df)} records")

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
                else:
                    # No date filtering applied - use all data
                    data_processor.sales_df = data_processor.df[data_processor.df['Tran Type'] == 'Sales'].copy()
                    data_processor.returns_df = data_processor.df[data_processor.df['Tran Type'] == 'Return'].copy()

                    # Clean return values for full data
                    if not data_processor.returns_df.empty:
                        data_processor.returns_df['Value'] = data_processor.returns_df['Value'].astype(str).str.replace(r'[()]', '', regex=True).astype(float)
                        data_processor.returns_df['Qty'] = data_processor.returns_df['Qty'].abs()

                    # Regenerate insights with full data
                    data_processor._generate_insights()

                # RAG System
                if RAG_AVAILABLE:
                    st.subheader("🤖 AI Assistant")
                    # API Key input with better handling
                    default_api_key = os.getenv('OPENROUTER_API_KEY', '')
                    api_key = st.text_input(
                        "OpenRouter API Key",
                        type="password",
                        help="Get your API key from https://openrouter.ai/",
                        value=default_api_key,
                        placeholder="sk-or-v1-..."
                    )

                    if api_key and len(api_key.strip()) > 10:  # Basic validation
                        try:
                            rag_system = RAGSystem(api_key)
                            context = rag_system.prepare_documents(data_processor)
                            if rag_system.is_initialized:
                                st.success("🤖 AI Assistant ready! Context prepared with comprehensive dashboard data.")
                                st.info(f"📊 Context includes: {len(rag_system.context.split())} words of detailed analysis")

                                # Debug info for troubleshooting
                                with st.expander("🔧 Debug Information"):
                                    st.write(f"**API Key Length:** {len(api_key)} characters")
                                    st.write(f"**API Key Format:** {'✅ Valid format' if api_key.startswith('sk-or-v1-') else '❌ Invalid format'}")
                                    st.write(f"**Context Size:** {len(rag_system.context)} characters")
                                    st.write(f"**Context Preview:** {rag_system.context[:200]}...")
                            else:
                                st.warning("⚠️ AI Assistant context preparation failed. Please check your data.")
                                st.info("💡 The AI assistant needs valid data to prepare context. Make sure your CSV file is loaded correctly.")
                        except Exception as e:
                            st.error(f"❌ AI Assistant initialization failed: {str(e)}")
                            st.info("💡 Make sure your OpenRouter API key is valid and has sufficient credits.")
                            with st.expander("🔧 Debug API Key"):
                                st.write(f"**API Key Provided:** {api_key[:20]}...")
                                st.write(f"**Key Format Check:** {'✅ Starts with sk-or-v1-' if api_key.startswith('sk-or-v1-') else '❌ Invalid format'}")

                                # Test API key with a simple request
                                if st.button("🧪 Test API Key", key="test_api_key"):
                                    with st.spinner("Testing API key..."):
                                        try:
                                            import requests
                                            test_headers = {
                                                "Authorization": f"Bearer {api_key}",
                                                "Content-Type": "application/json"
                                            }
                                            test_response = requests.get(
                                                "https://openrouter.ai/api/v1/auth/key",
                                                headers=test_headers,
                                                timeout=10
                                            )
                                            if test_response.status_code == 200:
                                                st.success("✅ API key is valid!")
                                                key_info = test_response.json()
                                                if 'data' in key_info:
                                                    st.info(f"📊 Credits remaining: ${key_info['data'].get('credits', 'Unknown')}")
                                            else:
                                                st.error(f"❌ API key test failed: {test_response.status_code} - {test_response.text}")
                                        except Exception as test_e:
                                            st.error(f"❌ API key test error: {str(test_e)}")

                                st.write("**Troubleshooting:**")
                                st.write("1. Verify the API key is copied correctly from OpenRouter")
                                st.write("2. Check if your OpenRouter account has credits")
                                st.write("3. Ensure the API key has not expired")
                                st.write("4. Try generating a new API key from OpenRouter")
                    else:
                        st.info("🔑 Enter your OpenRouter API key to enable AI assistant features.")


    # Main content
    if not hasattr(data_processor, 'df') or data_processor.df is None:
        st.info("👆 Please upload your vzrm_6.csv file to begin analysis.")
        return

    # Key Metrics Row (showing current filtered dataset totals)
    insights = data_processor.get_insights()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Show current dataset metrics (filtered or full based on date selection)
        # Using number of transactions as "Total Net Items Sold"
        total_transactions = insights.get('total_sales', 0)
        total_returns = insights.get('total_returns', 0)
        net_items = total_transactions - total_returns
        st.metric("Total Net Items Sold", f"{format_indian_number(net_items)}", f"Sales: {format_indian_number(total_transactions)}")

    with col2:
        # Show current dataset metrics (filtered or full based on date selection)
        net_revenue = insights.get('net_revenue', 0)
        total_revenue = insights.get('total_revenue', 0)
        st.metric("Total Net Revenue", f"₹{format_indian_number(net_revenue)}", f"Sales: ₹{format_indian_number(total_revenue)}")

    with col3:
        # Show current dataset unique products (filtered or full based on date selection)
        unique_products = insights.get('unique_products', 0)
        st.metric("Unique Products Sold", f"{unique_products}")

    with col4:
        # Calculate return rate based on current dataset (filtered or full based on date selection)
        current_total_sales = insights.get('total_sales', 0)
        current_total_returns = insights.get('total_returns', 0)
        actual_return_rate = (current_total_returns / current_total_sales * 100) if current_total_sales > 0 else 0
        st.metric("Overall Return Rate", f"{actual_return_rate:.2f}% ({format_indian_number(current_total_returns)} returns)")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏆 Performance Analysis",
        "📊 Brand Analysis",
        "📈 Product Analysis",
        "💰 Price & Discount Analysis",
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
        create_price_discount_analysis(data_processor)

    with tab5:
        create_returns_analysis(data_processor)

    with tab6:
        create_ai_insights(data_processor, rag_system)

    with tab7:
        create_business_insights(data_processor)

    with tab8:
        create_data_overview(data_processor)

def create_performance_analysis(data_processor: DataProcessor):
    """Create performance analysis tab"""
    st.markdown('<div class="section-header">🏆 Product Performance Analysis</div>', unsafe_allow_html=True)

    insights = data_processor.get_insights()

    # Sophisticated Multi-Level Filtering System
    st.markdown("### 🎯 Advanced Filtering")

    col_filter1, col_filter2 = st.columns([1, 1])

    with col_filter1:
        # Brand Selection
        available_brands = sorted(data_processor.sales_df['Brand Code'].unique()) if data_processor.sales_df is not None else []
        selected_brand = st.selectbox(
            "Select Brand:",
            options=["All Brands"] + available_brands,
            index=0,
            key="filter_brand"
        )

    with col_filter2:
        # Category Selection (dependent on brand)
        if selected_brand != "All Brands":
            brand_data = data_processor.sales_df[data_processor.sales_df['Brand Code'] == selected_brand]
            available_categories = get_product_categories(brand_data)
        else:
            available_categories = get_product_categories(data_processor.sales_df)

        selected_category = st.selectbox(
            "Select Category:",
            options=["All Categories"] + list(available_categories.keys()),
            index=0,
            key="filter_category"
        )

    # Color and Size filters (dependent on brand and category)
    col_filter3, col_filter4 = st.columns([1, 1])

    with col_filter3:
        # Color Selection
        if selected_brand != "All Brands" and selected_category != "All Categories":
            filtered_data = get_filtered_data(data_processor.sales_df, selected_brand, selected_category)
            available_colors = sorted(filtered_data['Shade Code'].dropna().unique()) if 'Shade Code' in filtered_data.columns else []
        elif selected_brand != "All Brands":
            filtered_data = data_processor.sales_df[data_processor.sales_df['Brand Code'] == selected_brand]
            available_colors = sorted(filtered_data['Shade Code'].dropna().unique()) if 'Shade Code' in filtered_data.columns else []
        else:
            available_colors = []

        selected_color = st.selectbox(
            "Select Color:",
            options=["All Colors"] + available_colors,
            index=0,
            key="filter_color"
        )

    with col_filter4:
        # Size Selection
        if selected_brand != "All Brands" and selected_category != "All Categories" and selected_color != "All Colors":
            filtered_data = get_filtered_data(data_processor.sales_df, selected_brand, selected_category, selected_color)
            available_sizes = sorted(filtered_data['Size'].dropna().unique()) if 'Size' in filtered_data.columns else []
        elif selected_brand != "All Brands" and selected_category != "All Categories":
            filtered_data = get_filtered_data(data_processor.sales_df, selected_brand, selected_category)
            available_sizes = sorted(filtered_data['Size'].dropna().unique()) if 'Size' in filtered_data.columns else []
        elif selected_brand != "All Brands":
            filtered_data = data_processor.sales_df[data_processor.sales_df['Brand Code'] == selected_brand]
            available_sizes = sorted(filtered_data['Size'].dropna().unique()) if 'Size' in filtered_data.columns else []
        else:
            available_sizes = []

        selected_size = st.selectbox(
            "Select Size:",
            options=["All Sizes"] + available_sizes,
            index=0,
            key="filter_size"
        )

    # Apply all filters to get the final filtered dataframe
    filtered_df = get_filtered_data(data_processor.sales_df, selected_brand, selected_category, selected_color, selected_size)

    # Show filter summary
    filter_summary = []
    if selected_brand != "All Brands":
        filter_summary.append(f"Brand: {get_brand_name(selected_brand)}")
    if selected_category != "All Categories":
        filter_summary.append(f"Category: {selected_category}")
    if selected_color != "All Colors":
        filter_summary.append(f"Color: {selected_color}")
    if selected_size != "All Sizes":
        filter_summary.append(f"Size: {selected_size}")

    if filter_summary:
        st.info(f"📊 Applied Filters: {', '.join(filter_summary)}")

    # Configurable top N selector for all graphs
    top_n = st.selectbox(
        "Select number of products to display:",
        options=[5, 10, 15, 20, 25, 50],
        index=1,  # 10 is at index 1
        key="product_performance_top_n"
    )

    # Combined Fastest & Best Selling Analysis
    st.subheader(f"Top {top_n} Fastest & Best Selling Products")

    sales_df = filtered_df if not filtered_df.empty else (data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame())
    returns_df = data_processor.returns_df if data_processor.returns_df is not None else pd.DataFrame()

    if not sales_df.empty:
        # Calculate daily rates for fastest selling - only use valid dates
        valid_date_sales = sales_df[~sales_df['Bill Date'].isna()]

        if not valid_date_sales.empty:
            min_date = valid_date_sales['Bill Date'].min()
            max_date = valid_date_sales['Bill Date'].max()
            days_in_period = (max_date - min_date).days + 1

            daily_rates = valid_date_sales.groupby(['Brand Code', 'Product Code']).agg({
                'Qty': 'sum'
            }).reset_index()

            daily_rates['Qty_Per_Day'] = (daily_rates['Qty'] / days_in_period).round(0).astype(int)
            daily_rates['Brand_Product'] = daily_rates['Brand Code'] + ' - ' + daily_rates['Product Code']
            daily_rates = daily_rates.sort_values('Qty_Per_Day', ascending=False)
            top_fastest = daily_rates.head(top_n)
        else:
            # If no valid dates, use all data but with a default period
            st.warning("⚠️ No valid dates for calculating daily rates. Using total quantity instead.")
            daily_rates = sales_df.groupby(['Brand Code', 'Product Code']).agg({
                'Qty': 'sum'
            }).reset_index()
            daily_rates['Qty_Per_Day'] = daily_rates['Qty']  # Use total quantity as daily rate
            daily_rates['Brand_Product'] = daily_rates['Brand Code'] + ' - ' + daily_rates['Product Code']
            daily_rates = daily_rates.sort_values('Qty_Per_Day', ascending=False)
            top_fastest = daily_rates.head(top_n)

        # Calculate best selling (net quantity)
        sales_summary = sales_df.groupby(['Brand Code', 'Product Code']).agg({
            'Qty': 'sum',
            'Value': 'sum'
        }).reset_index()

        if not returns_df.empty:
            returns_summary = returns_df.groupby(['Brand Code', 'Product Code']).agg({
                'Qty': lambda x: abs(x.sum()),
                'Value': lambda x: abs(x.sum())
            }).reset_index()
            returns_summary.columns = ['Brand Code', 'Product Code', 'Return_Qty', 'Return_Value']

            net_data = pd.merge(sales_summary, returns_summary[['Brand Code', 'Product Code', 'Return_Qty', 'Return_Value']],
                              on=['Brand Code', 'Product Code'], how='left')
            net_data[['Return_Qty', 'Return_Value']] = net_data[['Return_Qty', 'Return_Value']].fillna(0)
            net_data['Net_Qty'] = (net_data['Qty'] - net_data['Return_Qty']).astype(int)
        else:
            net_data = sales_summary.copy()
            net_data['Net_Qty'] = net_data['Qty'].astype(int)

        net_data['Brand_Product'] = net_data['Brand Code'] + ' - ' + net_data['Product Code']
        net_data = net_data.sort_values('Net_Qty', ascending=False)
        top_best = net_data.head(top_n)

        # Create combined dataframe for visualization
        combined_data = []
        for i, (_, row) in enumerate(top_fastest.iterrows()):
            combined_data.append({
                'Rank': i + 1,
                'Product': row['Brand_Product'],
                'Fastest_Qty_Day': row['Qty_Per_Day'],
                'Best_Net_Qty': 0,
                'Type': 'Fastest'
            })

        for i, (_, row) in enumerate(top_best.iterrows()):
            # Check if this product is already in fastest list
            existing = next((item for item in combined_data if item['Product'] == row['Brand_Product']), None)
            if existing:
                existing['Best_Net_Qty'] = row['Net_Qty']
            else:
                combined_data.append({
                    'Rank': i + 1,
                    'Product': row['Brand_Product'],
                    'Fastest_Qty_Day': 0,
                    'Best_Net_Qty': row['Net_Qty'],
                    'Type': 'Best'
                })

        combined_df = pd.DataFrame(combined_data)

        # Create grouped bar chart
        fig_combined = px.bar(
            combined_df,
            x='Product',
            y=['Fastest_Qty_Day', 'Best_Net_Qty'],
            title=f"Top {top_n} Fastest & Best Selling Products",
            barmode='group',
            color_discrete_sequence=['#2E86AB', '#A23B72'],
            height=max(500, top_n * 20)
        )

        # Customize the chart
        fig_combined.update_layout(
            xaxis_title="Products",
            yaxis_title="Quantity",
            legend_title="Metric Type",
            xaxis={'tickangle': -45}
        )

        # Update legend labels
        fig_combined.update_traces(
            name="Daily Quantity (Fastest)",
            selector=dict(name="Fastest_Qty_Day")
        )
        fig_combined.update_traces(
            name="Net Quantity (Best)",
            selector=dict(name="Best_Net_Qty")
        )

        st.plotly_chart(fig_combined, use_container_width=True)

    # Bottom performers section
    if not net_data.empty:
        st.subheader("⚠️ Bottom 5 Products (Need Attention)")

        bottom_n = st.selectbox(
            "Select number of products to display:",
            options=[5, 10, 15, 20, 25, 50],
            index=0,  # 5 is at index 0
            key="bottom_products_n"
        )

        bottom_products = net_data.nsmallest(bottom_n, 'Net_Qty')
        bottom_products['Brand_Product'] = bottom_products['Brand Code'] + ' - ' + bottom_products['Product Code']

        fig_bottom = px.bar(
            bottom_products,
            x='Net_Qty',
            y='Brand_Product',
            title=f"Bottom {bottom_n} Products by Net Quantity",
            color='Net_Qty',
            color_continuous_scale='reds',
            orientation='h'
        )
        fig_bottom.update_layout(height=max(400, bottom_n * 15), yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bottom, use_container_width=True)

    # Size and Color Analysis
    st.markdown("**Size & Color Analysis**")


    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Most Popular Sizes**")

        if not sales_df.empty and 'Size' in sales_df.columns:
            size_analysis = sales_df.groupby('Size').agg({
                'Qty': 'sum',
                'Value': 'sum'
            }).reset_index()

            # Filter out empty or invalid sizes
            size_analysis = size_analysis[size_analysis['Size'].notna() & (size_analysis['Size'] != '')]

            if not size_analysis.empty:
                size_analysis = size_analysis.nlargest(top_n, 'Qty')

                fig_sizes = px.bar(
                    size_analysis,
                    x='Qty',
                    y='Size',
                    title=f"Top {top_n} Most Popular Sizes",
                    color='Qty',
                    color_continuous_scale='blues',
                    orientation='h'
                )
                fig_sizes.update_layout(height=max(400, len(size_analysis) * 15), yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_sizes, use_container_width=True)

    with col4:
        st.markdown("**Color Analysis**")

        if not sales_df.empty and 'Shade Code' in sales_df.columns:
            color_analysis = sales_df.groupby(['Brand Code', 'Shade Code']).agg({
                'Qty': 'sum',
                'Value': 'sum'
            }).reset_index()

            # Filter out empty or invalid colors
            color_analysis = color_analysis[color_analysis['Shade Code'].notna() & (color_analysis['Shade Code'] != '')]

            if not color_analysis.empty:
                # Sort by quantity and get top N
                color_analysis = color_analysis.nlargest(top_n, 'Qty')

                # Add brand names
                color_analysis['Brand Name'] = color_analysis['Brand Code'].apply(get_brand_name)

                fig_colors = px.bar(
                    color_analysis,
                    x='Qty',
                    y='Shade Code',
                    title=f"Top {top_n} Most Popular Colors",
                    color='Qty',
                    color_continuous_scale='rainbow',
                    orientation='h',
                    hover_data=['Brand Name', 'Brand Code']
                )
                fig_colors.update_layout(height=max(400, len(color_analysis) * 15), yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_colors, use_container_width=True)

                # Show detailed color-brand breakdown
                st.write("**Color-Brand Breakdown:**")
                display_colors = color_analysis.copy()
                display_colors['Total_Quantity'] = display_colors['Qty'].astype(int)
                display_colors['Total_Value'] = display_colors['Value'].apply(lambda x: f"₹{format_indian_number(int(x))}")

                st.dataframe(display_colors[['Brand Name', 'Brand Code', 'Shade Code', 'Total_Quantity', 'Total_Value']])

                # Size-Color combinations (most popular)
                if 'Size' in sales_df.columns:
                    size_color_analysis = sales_df.groupby(['Size', 'Shade Code', 'Brand Code']).agg({
                        'Qty': 'sum'
                    }).reset_index()

                    size_color_analysis = size_color_analysis[size_color_analysis['Size'].notna() & (size_color_analysis['Size'] != '')]
                    size_color_analysis = size_color_analysis.nlargest(10, 'Qty')

                    if not size_color_analysis.empty:
                        st.write("**Top Size-Color Combinations:**")
                        display_size_color = size_color_analysis.copy()
                        display_size_color['Brand Name'] = display_size_color['Brand Code'].apply(get_brand_name)
                        display_size_color['Total_Quantity'] = display_size_color['Qty'].astype(int)

                        st.dataframe(display_size_color[['Brand Name', 'Brand Code', 'Size', 'Shade Code', 'Total_Quantity']].head(10))

    # Product Category Analysis
    st.markdown("**Product Category Analysis**")

    if not sales_df.empty:
        # Analyze product categories based on product codes
        category_keywords = {
            'T-Shirts': ['T-SHIRT', 'TEE', 'T SHIRT'],
            'Shirts': ['SHIRT'],
            'Trousers': ['TROUSER', 'PANT'],
            'Jeans': ['JEANS', 'DENIM'],
            'Jackets': ['JACKET', 'BLAZER'],
            'Others': []
        }

        category_data = []
        for category, keywords in category_keywords.items():
            if keywords:
                # Filter products that match this category
                mask = sales_df['Product Code'].str.contains('|'.join(keywords), case=False, na=False)
                category_sales = sales_df[mask]
            else:
                # For 'Others', get products that don't match any other category
                other_mask = pd.Series([True] * len(sales_df), index=sales_df.index)
                for other_keywords in [v for k, v in category_keywords.items() if v]:
                    other_mask &= ~sales_df['Product Code'].str.contains('|'.join(other_keywords), case=False, na=False)
                category_sales = sales_df[other_mask]

            if not category_sales.empty:
                total_qty = int(category_sales['Qty'].sum())
                total_value = category_sales['Value'].sum()
                category_data.append({
                    'Category': category,
                    'Total_Quantity': total_qty,
                    'Total_Value': total_value,
                    'Average_Price': total_value / total_qty if total_qty > 0 else 0
                })

        if category_data:
            category_df = pd.DataFrame(category_data).sort_values('Total_Quantity', ascending=False)

            # Create chart for product categories
            fig_categories = px.bar(
                category_df,
                x='Category',
                y='Total_Quantity',
                title="Product Categories by Quantity Sold",
                color='Total_Quantity',
                color_continuous_scale='viridis'
            )
            fig_categories.update_layout(height=400)
            st.plotly_chart(fig_categories, use_container_width=True)

            # Show category breakdown table
            display_df = category_df.copy()
            display_df['Total_Quantity'] = display_df['Total_Quantity'].apply(format_indian_number)
            display_df['Total_Value'] = display_df['Total_Value'].apply(lambda x: f"₹{format_indian_number(int(x))}")
            display_df['Average_Price'] = display_df['Average_Price'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display_df[['Category', 'Total_Quantity', 'Total_Value', 'Average_Price']])

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

        # Apply the same filtering logic as performance analysis
        selected_brand = st.session_state.get('filter_brand', 'All Brands')
        selected_category = st.session_state.get('filter_category', 'All Categories')
        selected_color = st.session_state.get('filter_color', 'All Colors')
        selected_size = st.session_state.get('filter_size', 'All Sizes')

        filtered_sales_df = get_filtered_data(sales_df, selected_brand, selected_category, selected_color, selected_size)
        filtered_returns_df = get_filtered_data(returns_df, selected_brand, selected_category, selected_color, selected_size)

        sales_df = filtered_sales_df if not filtered_sales_df.empty else sales_df
        returns_df = filtered_returns_df if not filtered_returns_df.empty else returns_df

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
                brand_analysis['Net_Qty'] = (brand_analysis['Qty'] - brand_analysis['Return_Qty']).astype(int)
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
                brand_analysis['Net_Qty'] = brand_analysis['Qty'].astype(int)
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
                st.plotly_chart(fig_brand_qty, use_container_width=True)

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
            st.plotly_chart(fig_brand_rev, use_container_width=True)

    # Most Reliable Brands
    if not brand_analysis.empty:
        st.subheader("🌟 Most Reliable Brands")

        # Configurable top N selector for reliable brands
        top_n_reliable = st.selectbox(
            "Select number of brands to display:",
            options=[5, 10, 15, 20, 25],
            index=1,  # 10 is at index 1
            key="reliable_brands_n"
        )

        reliable_brands = brand_analysis.sort_values('Reliability_Score', ascending=False)
        reliable_brands['Brand Name'] = reliable_brands['Brand Code'].apply(get_brand_name)

        st.dataframe(reliable_brands[['Brand Name', 'Brand Code', 'Net_Qty', 'Return_Rate', 'CV', 'Reliability_Score']].head(top_n_reliable))

def create_product_analysis(data_processor: DataProcessor):
    """Create product analysis tab"""
    st.markdown('<div class="section-header">📈 Product Analysis</div>', unsafe_allow_html=True)

    # Use filtered data if available
    sales_df = data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame()
    returns_df = data_processor.returns_df if data_processor.returns_df is not None else pd.DataFrame()

    # Apply the same filtering logic as performance analysis
    # Get filter values from session state
    selected_brand = st.session_state.get('filter_brand', 'All Brands')
    selected_category = st.session_state.get('filter_category', 'All Categories')
    selected_color = st.session_state.get('filter_color', 'All Colors')
    selected_size = st.session_state.get('filter_size', 'All Sizes')

    filtered_df = get_filtered_data(sales_df, selected_brand, selected_category, selected_color, selected_size)

    # Calculate insights based on filtered data
    if not filtered_df.empty:
        # Group sales by product for top products
        product_performance = filtered_df.groupby(['Brand Code', 'Product Code']).agg({
            'Qty': 'sum',
            'Value': 'sum'
        }).reset_index()

        top_products_qty = product_performance.nlargest(50, 'Qty')[['Brand Code', 'Product Code', 'Qty']].to_dict('records')
        top_products_value = product_performance.nlargest(50, 'Value')[['Brand Code', 'Product Code', 'Value']].to_dict('records')
    else:
        top_products_qty = []
        top_products_value = []

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥇 Top Products by Quantity")

        # Configurable top N selector
        top_n_qty = st.selectbox(
            "Select number of products to display:",
            options=[5, 10, 15, 20, 25, 50],
            index=1,  # 10 is at index 1
            key="top_products_qty_n"
        )

        if top_products_qty:
            products_df = pd.DataFrame(top_products_qty)
            products_df['Brand_Product'] = products_df['Brand Code'] + ' - ' + products_df['Product Code']
            top_products_qty_display = products_df.head(top_n_qty)
            fig = px.bar(top_products_qty_display, x='Qty', y='Brand_Product', orientation='h', title=f"Top {top_n_qty} Products by Quantity")
            fig.update_layout(height=max(400, top_n_qty * 15), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💎 Top Products by Revenue")

        # Configurable top N selector
        top_n_value = st.selectbox(
            "Select number of products to display:",
            options=[5, 10, 15, 20, 25, 50],
            index=1,  # 10 is at index 1
            key="top_products_value_n"
        )

        if top_products_value:
            products_value_df = pd.DataFrame(top_products_value)
            products_value_df['Brand_Product'] = products_value_df['Brand Code'] + ' - ' + products_value_df['Product Code']
            top_products_value_display = products_value_df.head(top_n_value)
            fig = px.bar(top_products_value_display, x='Value', y='Brand_Product', orientation='h', title=f"Top {top_n_value} Products by Revenue")
            fig.update_layout(height=max(400, top_n_value * 15), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

def create_price_discount_analysis(data_processor: DataProcessor):
    """Create price and discount analysis tab"""
    st.markdown('<div class="section-header">💰 Price & Discount Analysis</div>', unsafe_allow_html=True)

    if not hasattr(data_processor, 'df') or data_processor.df is None or data_processor.df.empty:
        st.info("No data available for price analysis.")
        return

    df = data_processor.df.copy()

    # Filter for sales only
    sales_df = df[df['Tran Type'] == 'Sales'].copy()

    if sales_df.empty:
        st.info("No sales data available for price analysis.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Price Analysis**")

        # Calculate price metrics
        avg_mrp = sales_df['MRP'].mean()
        avg_selling_price = sales_df['Value'].mean()
        avg_discount = ((sales_df['MRP'] - sales_df['Value']) / sales_df['MRP'] * 100).mean()

        # Price distribution
        price_ranges = pd.cut(sales_df['Value'], bins=[0, 500, 1000, 2000, 5000, float('inf')],
                            labels=['₹0-500', '₹500-1000', '₹1000-2000', '₹2000-5000', '₹5000+'])

        price_dist = price_ranges.value_counts().sort_index()

        fig_price_dist = px.bar(
            x=price_dist.index,
            y=price_dist.values,
            title="Sales Distribution by Price Range",
            labels={'x': 'Price Range', 'y': 'Number of Items Sold'}
        )
        fig_price_dist.update_layout(height=400)
        st.plotly_chart(fig_price_dist, use_container_width=True)

        # Key metrics
        st.write("**Key Price Metrics:**")
        st.write(f"• Average MRP: ₹{avg_mrp:,.0f}")
        st.write(f"• Average Selling Price: ₹{avg_selling_price:,.0f}")
        st.write(f"• Average Discount: {avg_discount:.1f}%")

    with col2:
        st.markdown("**Discount Analysis**")

        # Calculate discount by brand
        discount_by_brand = sales_df.groupby('Brand Code').agg({
            'MRP': 'mean',
            'Value': 'mean',
            'Qty': 'sum'
        }).reset_index()

        discount_by_brand['Discount_Percent'] = ((discount_by_brand['MRP'] - discount_by_brand['Value']) / discount_by_brand['MRP'] * 100)
        discount_by_brand['Discount_Amount'] = discount_by_brand['MRP'] - discount_by_brand['Value']

        # Sort by discount percentage
        top_discount_brands = discount_by_brand.nlargest(10, 'Discount_Percent')

        fig_discount = px.bar(
            top_discount_brands,
            x='Discount_Percent',
            y='Brand Code',
            title="Top 10 Brands by Discount Percentage",
            orientation='h',
            color='Discount_Percent',
            color_continuous_scale='oranges'
        )
        fig_discount.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_discount, use_container_width=True)

        # Show discount impact table
        display_discount = top_discount_brands.copy()
        display_discount['Brand Name'] = display_discount['Brand Code'].apply(get_brand_name)
        display_discount['Discount_Percent'] = display_discount['Discount_Percent'].round(1)
        display_discount['Avg_Discount_Amount'] = display_discount['Discount_Amount'].round(0)
        display_discount['Total_Quantity'] = display_discount['Qty'].astype(int)

        st.dataframe(display_discount[['Brand Name', 'Brand Code', 'Discount_Percent', 'Avg_Discount_Amount', 'Total_Quantity']])

def create_returns_analysis(data_processor: DataProcessor):
    """Create returns analysis tab"""
    st.markdown('<div class="section-header">🔄 Returns Analysis</div>', unsafe_allow_html=True)

    # Use filtered data if available
    sales_df = data_processor.sales_df if data_processor.sales_df is not None else pd.DataFrame()
    returns_df = data_processor.returns_df if data_processor.returns_df is not None else pd.DataFrame()

    # Apply the same filtering logic as performance analysis
    selected_brand = st.session_state.get('filter_brand', 'All Brands')
    selected_category = st.session_state.get('filter_category', 'All Categories')
    selected_color = st.session_state.get('filter_color', 'All Colors')
    selected_size = st.session_state.get('filter_size', 'All Sizes')

    filtered_returns_df = get_filtered_data(returns_df, selected_brand, selected_category, selected_color, selected_size)

    # Calculate return analysis based on filtered data
    if not filtered_returns_df.empty:
        return_analysis = filtered_returns_df.groupby(['Brand Code', 'Product Code']).agg({
            'Qty': lambda x: abs(x.sum()),
            'Value': lambda x: abs(x.sum())
        }).reset_index()

        return_analysis = return_analysis.to_dict('records')
    else:
        return_analysis = []

    if not return_analysis:
        st.info("No returns data available for the selected filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Products with Most Returns")

        # Configurable top N selector
        top_n_returns_qty = st.selectbox(
            "Select number of products to display:",
            options=[5, 10, 15, 20, 25, 50],
            index=1,  # 10 is at index 1
            key="returns_qty_n"
        )

        returns_df = pd.DataFrame(return_analysis)
        returns_df['Brand_Product'] = returns_df['Brand Code'] + ' - ' + returns_df['Product Code']
        top_returns = returns_df.nlargest(top_n_returns_qty, 'Qty')
        fig = px.bar(top_returns, x='Qty', y='Brand_Product', orientation='h', title=f"Top {top_n_returns_qty} Products with Most Returns")
        fig.update_layout(height=max(400, top_n_returns_qty * 15), yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💸 Return Value Analysis")

        # Configurable top N selector
        top_n_returns_value = st.selectbox(
            "Select number of products to display:",
            options=[5, 10, 15, 20, 25, 50],
            index=1,  # 10 is at index 1
            key="returns_value_n"
        )

        top_returns_value = returns_df.nlargest(top_n_returns_value, 'Value')
        fig = px.bar(top_returns_value, x='Value', y='Brand_Product', orientation='h', title=f"Top {top_n_returns_value} Products with Highest Return Value")
        fig.update_layout(height=max(400, top_n_returns_value * 15), yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

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

        # Ensure all values are valid numbers before creating DataFrame
        total_transactions = perf_insights['key_metrics'].get('total_transactions', 0) or 0
        total_revenue = perf_insights['key_metrics'].get('total_revenue', 0) or 0
        net_revenue = perf_insights['key_metrics'].get('net_revenue', 0) or 0
        return_rate = perf_insights['key_metrics'].get('return_rate', 0) or 0

        metrics_df = pd.DataFrame([
            ["Total Transactions", f"{total_transactions:,}"],
            ["Total Revenue", f"₹{total_revenue:,.0f}"],
            ["Net Revenue", f"₹{net_revenue:,.0f}"],
            ["Return Rate", f"{return_rate:.2f}% ({format_indian_number(data_processor.get_insights().get('return_count', 0))} returns)"]
        ], columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)

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
        if customer_insights.get('size_demand'):
            size_demand = customer_insights['size_demand']
            st.write("**Size Demand Analysis:**")

            if size_demand['top_sizes']:
                st.write("**Top Size-Brand Combinations:**")
                for item in size_demand['top_sizes'][:5]:
                    st.write(f"• {item['Brand Code']} - Size {item['Size']}: {item['Qty']} units sold")

            if size_demand['brand_size_preferences']:
                st.write("**Brand Size Preferences:**")
                current_brand = None
                for pref in size_demand['brand_size_preferences'][:8]:
                    if current_brand != pref['Brand Code']:
                        if current_brand is not None:
                            st.write("")  # Add spacing between brands
                        current_brand = pref['Brand Code']
                        st.write(f"**{pref['Brand Code']}:**")
                    st.write(f"  - Size {pref['Size']}: {pref['Qty']} units")

            # Show size distribution chart
            if size_demand['size_distribution']:
                st.write("**Size Distribution:**")
                size_dist_df = pd.DataFrame(list(size_demand['size_distribution'].items()),
                                          columns=['Size', 'Quantity'])
                size_dist_df = size_dist_df.sort_values('Quantity', ascending=False).head(10)

                fig_sizes = px.bar(
                    size_dist_df,
                    x='Quantity',
                    y='Size',
                    title="Top 10 Most Demanded Sizes",
                    color='Quantity',
                    color_continuous_scale='blues',
                    orientation='h'
                )
                fig_sizes.update_layout(height=max(400, len(size_dist_df) * 20))
                st.plotly_chart(fig_sizes, use_container_width=True)

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

    # Sample data - use filtered data if available
    st.subheader("🔍 Sample Data")
    if data_processor.sales_df is not None and not data_processor.sales_df.empty:
        # Show sample of filtered data
        sample_data = data_processor.sales_df.head(10)
        if data_processor.returns_df is not None and not data_processor.returns_df.empty:
            returns_sample = data_processor.returns_df.head(5)
            sample_data = pd.concat([sample_data, returns_sample])
        st.dataframe(sample_data)
        st.caption("Showing sample of filtered data based on selected date range")
    else:
        # Show sample of full data
        st.dataframe(data_processor.df.head(20))

    # Download options
    st.subheader("💾 Export Data")

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    col1, col2, col3 = st.columns(3)

    with col1:
        if data_processor.sales_df is not None and not data_processor.sales_df.empty:
            # Show filtered data download options
            filtered_data = pd.concat([data_processor.sales_df, data_processor.returns_df]) if data_processor.returns_df is not None and not data_processor.returns_df.empty else data_processor.sales_df
            csv_filtered = convert_df(filtered_data)
            st.download_button(
                label="📄 Download Filtered Dataset",
                data=csv_filtered,
                file_name='tipsy_topsy_filtered_data.csv',
                mime='text/csv',
            )
        else:
            # Show full data download
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
