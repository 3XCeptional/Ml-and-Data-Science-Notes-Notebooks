Here's a comprehensive **Pandas Cheatsheet** designed for quick reference to common tasks in data science. It covers essential commands, handling data, and more advanced use cases, including those frequently encountered in data science workflows:

### **1. Importing Pandas & Loading Data**
```python
import pandas as pd

# Read CSV, Excel, or JSON file
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_json('file.json')

# Read from SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM table_name', conn)

# Save to file
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
```

### **2. Data Inspection**
```python
# First & Last 5 rows
df.head()  # First 5 rows
df.tail()  # Last 5 rows

# Data dimensions
df.shape  # (rows, columns)

# Column names
df.columns

# Data types
df.dtypes

# Basic stats summary
df.describe()

# Memory usage
df.info()
```

### **3. Selecting Data**
```python
# Select single column
df['column_name']  # or df.column_name

# Select multiple columns
df[['column1', 'column2']]

# Select rows by index
df.iloc[0]  # First row
df.iloc[0:5]  # First 5 rows

# Select rows by condition
df[df['column'] > 5]  # Rows where column value > 5
df[df['column'].str.contains('text')]  # String matching
```

### **4. Filtering & Sorting**
```python
# Filter data
df[df['Age'] > 30]  # Filter by condition
df.query('Age > 30 & Salary > 50000')  # Using query string

# Sort data
df.sort_values('column', ascending=False)

# Drop duplicates
df.drop_duplicates()

# Drop missing values (NaN)
df.dropna()

# Fill missing values (NaN)
df.fillna(value=0)
```

### **5. Adding & Removing Columns**
```python
# Add a new column
df['new_col'] = df['col1'] + df['col2']

# Remove a column
df.drop('column_name', axis=1, inplace=True)

# Rename columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### **6. Handling Missing Data**
```python
# Detect missing values
df.isna().sum()

# Drop rows with NaN values
df.dropna(subset=['column_name'])

# Fill missing data with a specific value
df['column_name'].fillna(value='Unknown', inplace=True)

# Fill missing data using method
df['column_name'].fillna(method='ffill')  # Forward fill
df['column_name'].fillna(method='bfill')  # Backward fill
```

### **7. Grouping & Aggregation**
```python
# Group by column(s) and aggregate
df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})

# Count occurrences
df['column'].value_counts()

# Pivot table
df.pivot_table(values='col1', index='col2', columns='col3', aggfunc='mean')

# Cross-tabulation (contingency table)
pd.crosstab(df['col1'], df['col2'])
```

### **8. Merging & Joining DataFrames**
```python
# Concatenate dataframes (vertically or horizontally)
pd.concat([df1, df2], axis=0)  # Vertically
pd.concat([df1, df2], axis=1)  # Horizontally

# Merge on common columns
pd.merge(df1, df2, on='common_column')

# Left, right, inner, and outer joins
pd.merge(df1, df2, how='left', on='column')
pd.merge(df1, df2, how='right', on='column')
pd.merge(df1, df2, how='inner', on='column')
pd.merge(df1, df2, how='outer', on='column')
```

### **9. Date & Time Handling**
```python
# Convert to datetime
df['date_column'] = pd.to_datetime(df['date_column'])

# Extract year, month, day
df['year'] = df['date_column'].dt.year
df['month'] = df['date_column'].dt.month
df['day'] = df['date_column'].dt.day

# Time difference between dates
df['diff_days'] = (df['end_date'] - df['start_date']).dt.days
```

### **10. String Manipulation**
```python
# Lowercase, Uppercase
df['col'] = df['col'].str.lower()
df['col'] = df['col'].str.upper()

# Remove whitespace
df['col'] = df['col'].str.strip()

# Replace values
df['col'] = df['col'].str.replace('old', 'new')

# Extract substrings
df['col'] = df['col'].str[:5]  # First 5 characters
df['col'] = df['col'].str.extract('(\d+)')  # Extract digits
```

### **11. Applying Functions**
```python
# Apply function to column
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Apply function to rows/columns
df.apply(np.sum, axis=0)  # Apply sum to columns
df.apply(np.sum, axis=1)  # Apply sum to rows

# Apply custom function
def custom_func(x):
    return x * 2
df['new_col'] = df['col'].apply(custom_func)
```

### **12. Reshaping Data**
```python
# Reshape with pivot
df.pivot(index='column1', columns='column2', values='value')

# Melt (unpivot) data
df_melted = pd.melt(df, id_vars=['column1'], value_vars=['column2', 'column3'])

# Transpose
df.T
```

### **13. Handling Large Datasets**
```python
# Load data in chunks
for chunk in pd.read_csv('file.csv', chunksize=10000):
    process(chunk)

# Optimize memory usage by specifying dtypes
df = pd.read_csv('file.csv', dtype={'col1': 'int32', 'col2': 'float32'})
```

### **14. Time Series Analysis**
```python
# Resampling time series data
df.set_index('date_column', inplace=True)
df.resample('M').mean()  # Resample by month and calculate mean

# Rolling window calculations
df['rolling_mean'] = df['value'].rolling(window=3).mean()
```

### **15. Exporting Data**
```python
# Save to CSV or Excel
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)

# Export to SQL
df.to_sql('table_name', conn, if_exists='replace')
```

This cheatsheet covers a wide range of common tasks and workflows encountered by data scientists when using Pandas. Keep it handy for your day-to-day data manipulations!

