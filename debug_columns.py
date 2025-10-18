import pandas as pd
import sys
sys.path.append('.')
from app.rag.vector_store import get_vector_store

print('=== CHECKING VECTOR STORE ===')
try:
    vector_store = get_vector_store()
    # Try to list collections - this might not work with Chroma
    print('Vector store initialized successfully')
except Exception as e:
    print(f'Error initializing vector store: {e}')

print('\n=== CHECKING EXCEL FILE ===')
df = pd.read_excel('BVRM.xlsx')
print('Actual Excel columns:')
for i, col in enumerate(df.columns):
    print(f'{i+1:2d}: "{col}"')
print(f'\nTotal columns: {len(df.columns)}')
print(f'Total rows: {len(df)}')

# Check column names after stripping
stripped_cols = df.columns.str.strip()
print(f'\nColumns after stripping spaces: {list(stripped_cols)}')

# Check if 'Brand Code' exists after stripping
brand_col_exists = 'Brand Code' in stripped_cols
print(f'\n"Brand Code" column exists after stripping: {brand_col_exists}')

# Check similar column names
similar_cols = [col for col in stripped_cols if 'brand' in col.lower() or 'code' in col.lower()]
print(f'Columns with "brand" or "code" after stripping: {similar_cols}')

print(f'\nValue column range: {df["Value"].min()} to {df["Value"].max()}')
print(f'Sample data:')
print(df[['Bill Date', 'Tran Type', 'Value']].head(3))
