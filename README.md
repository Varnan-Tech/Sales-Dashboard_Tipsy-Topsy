# Tipsy Topsy Analytics Dashboard

A comprehensive business intelligence dashboard for clothing store analytics with AI-powered insights and RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Features

### ✅ Unified Data Processing
- **Multiple File Format Support**: Works with CSV (.csv), Excel (.xlsx), and Excel Binary (.xlsb) files
- **Single File Input**: Works with your existing `vzrm_6.csv` file (no need for separate sales/returns files)
- **Automatic Data Cleaning**: Handles mixed sales and returns data automatically
- **Smart Column Processing**: Converts dates, numbers, and handles edge cases across all formats

### 📊 Advanced Analytics
- **Performance Analysis**: Track sales trends, top performers, and growth patterns
- **Brand Analysis**: Deep dive into brand performance and reliability metrics
- **Product Analysis**: Identify best-selling products and underperformers
- **Returns Analysis**: Understand return patterns and identify problematic products
- **Customer Insights**: Analyze size preferences and purchase patterns

### 🤖 AI-Powered Features
- **RAG System**: Retrieval-Augmented Generation using OpenAI via OpenRouter
- **Chatbot Interface**: Ask natural language questions about your data
- **Smart Insights**: AI-generated business recommendations
- **Automated Analysis**: Get insights without manual data exploration

### 💡 Business Intelligence
- **Actionable Recommendations**: Priority-based suggestions for business improvement
- **Trend Analysis**: Identify seasonal patterns and growth opportunities
- **Performance Scoring**: Brand reliability and product performance metrics
- **Export Capabilities**: Download insights and reports

## 📋 System Requirements

### Prerequisites
- Python 3.8+
- Data file: `vzrm_6.csv`, `.xlsx`, or `.xlsb` (your existing sales data file)
- OpenRouter API key (for AI features)

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: The requirements include packages for Excel file support (`openpyxl` for .xlsx, `pyxlsb` for .xlsb files).

2. **Verify Installation**
```bash
python test_system.py
```

## 🎯 Quick Start

### Option 1: Using the Dashboard (Recommended)

1. **Run the Dashboard**
```bash
streamlit run app.py
```

2. **Upload Your Data**
   - Open the dashboard in your browser
   - Upload your data file in the sidebar (supports CSV, XLSX, XLSB)
   - The system will automatically process mixed sales/returns data
   - For Excel files, ensure the first sheet contains your data

3. **Enable AI Features** (Optional)
   - Get an API key from [OpenRouter](https://openrouter.ai/)
   - Enter the API key in the sidebar
   - Start asking questions about your data

### Option 2: Using Individual Scripts

```bash
# Process your vzrm_6.csv file
python sales.py    # Creates sales_data_cleaned.csv
python return.py   # Creates returns_data_cleaned.csv
python new.py      # Runs original dashboard
```

## 📊 Dashboard Features

### 🏆 Performance Analysis
- Daily sales trends and revenue tracking
- Sales vs returns comparison
- Growth rate analysis

### 📈 Brand Analysis
- Top performing brands by quantity and revenue
- Brand reliability scoring (return rate + consistency)
- Brand concentration analysis

### 👕 Product Analysis
- Best-selling products identification
- Product performance by category
- Size distribution analysis

### 🔄 Returns Analysis
- Products with highest return rates
- Return value analysis
- Return pattern identification

### 🤖 AI Insights
- **Chat Interface**: Ask questions like:
  - "What are my top 5 best-selling products?"
  - "Which brands have the lowest return rates?"
  - "What should I do to reduce returns?"
  - "Show me sales trends for the last week"

### 💡 Business Insights
- Automated performance assessment
- Priority-based recommendations
- Actionable improvement suggestions
- Exportable business reports

## 🔧 Data Format

Your `vzrm_6.csv` file should contain these columns:

| Column | Description | Type |
|--------|-------------|------|
| Bill Date | Transaction date (DD/MM/YYYY) | Date |
| Tran Type | "Sales" or "Return" | String |
| Product Code | Product identifier | String |
| Brand Code | Brand identifier | String |
| Qty | Quantity | Number |
| Value | Transaction value | Number |
| Size | Product size | String |
| Cost | Cost price | Number |
| MRP | Maximum retail price | Number |

## 🎛️ Configuration

### Environment Variables
Create a `.env` file (optional):
```env
OPENROUTER_API_KEY=your_api_key_here
```

### API Configuration
- **OpenRouter**: Used for AI features (chatbot, insights)
- **No Local Models**: Everything runs through cloud APIs

## 📈 What You Can Do With This Data

### Business Questions Answered
1. **Performance**: "How is my business performing this month?"
2. **Products**: "Which products should I stock more of?"
3. **Brands**: "Which brands are most reliable?"
4. **Returns**: "Why are customers returning products?"
5. **Trends**: "What are the seasonal patterns in my sales?"
6. **Growth**: "How can I improve my business?"

### Key Insights Generated
- **Top Performers**: Best-selling products and brands
- **Problem Areas**: High-return products and brands
- **Growth Opportunities**: Trending products and untapped markets
- **Risk Assessment**: Brand concentration and reliability
- **Customer Preferences**: Size popularity and purchase patterns

## 🚀 Advanced Usage

### Custom Analysis
The system can be extended to:
- Add new KPIs and metrics
- Integrate with other data sources
- Create custom dashboards
- Set up automated reporting

### AI Integration
- Fine-tune prompts for specific business needs
- Add domain-specific knowledge
- Integrate with other AI services

## 🛠️ Troubleshooting

### Common Issues

1. **File Upload Issues**
   - Ensure `vzrm_6.csv` is in the correct format
   - Check for encoding issues (should be UTF-8)

2. **AI Features Not Working**
   - Verify OpenRouter API key is valid
   - Check internet connection
   - Ensure sufficient API credits

3. **Performance Issues**
   - Large files may take time to process
   - Consider filtering data for faster analysis

### Support
- Check the test script: `python test_system.py`
- Verify all dependencies are installed
- Ensure Python 3.8+ is being used

## 📝 License

This project is designed for Tipsy Topsy business analytics. All rights reserved.

## 🔄 Updates

The system automatically processes your latest `vzrm_6.csv` file each time you upload it, so you always get the most current insights and analysis.

---

**Ready to transform your business data into actionable insights?** 🎯
