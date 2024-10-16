Here's a **Seaborn Cheatsheet** that covers essential commands and visualizations for data scientists. Seaborn is a powerful Python library built on top of Matplotlib that simplifies statistical data visualization, especially when working with Pandas DataFrames.

---

### **1. Importing Seaborn**
```python
import seaborn as sns
import matplotlib.pyplot as plt  # Seaborn integrates well with Matplotlib
```

### **2. Built-in Datasets**
Seaborn provides several built-in datasets that are great for practicing.
```python
# Load a dataset
df = sns.load_dataset('tips')

# View the first few rows
df.head()
```

---

### **3. Setting Seaborn Styles**
Seaborn allows you to control the aesthetic of your plots.
```python
# Set the default Seaborn style
sns.set(style='whitegrid')

# Other options: 'darkgrid', 'white', 'dark', 'ticks'
```

---

### **4. Basic Plots**

#### **Scatter Plot**
For visualizing the relationship between two continuous variables.
```python
# Scatter plot with regression line
sns.lmplot(x='total_bill', y='tip', data=df, ci=None)

# Without regression line
sns.scatterplot(x='total_bill', y='tip', data=df)
```

#### **Line Plot**
For visualizing data over continuous values.
```python
# Line plot with markers
sns.lineplot(x='size', y='tip', data=df, marker='o')
```

#### **Bar Plot**
For visualizing aggregate data with categorical variables.
```python
# Basic bar plot
sns.barplot(x='day', y='total_bill', data=df)

# Grouped bar plot
sns.barplot(x='day', y='total_bill', hue='sex', data=df)
```

#### **Histogram (Distribution Plot)**
For visualizing the distribution of a single variable.
```python
# Simple histogram
sns.histplot(df['total_bill'], kde=False, bins=20)

# Add Kernel Density Estimate (KDE)
sns.histplot(df['total_bill'], kde=True)
```

#### **Kernel Density Plot (KDE)**
For showing the probability density of a variable.
```python
sns.kdeplot(df['total_bill'], shade=True)
```

#### **Count Plot**
For visualizing the frequency of categorical variables.
```python
sns.countplot(x='day', data=df)

# Count plot with hue
sns.countplot(x='day', hue='sex', data=df)
```

---

### **5. Categorical Plotting**

#### **Box Plot**
For visualizing the distribution of a dataset and detecting outliers.
```python
sns.boxplot(x='day', y='total_bill', data=df)

# Grouped box plot
sns.boxplot(x='day', y='total_bill', hue='sex', data=df)
```

#### **Violin Plot**
Similar to a box plot but also shows the probability density.
```python
sns.violinplot(x='day', y='total_bill', data=df)

# Split violins by category
sns.violinplot(x='day', y='total_bill', hue='sex', split=True, data=df)
```

#### **Swarm Plot**
For visualizing the distribution of data points along with categorical variables (without overlap).
```python
sns.swarmplot(x='day', y='total_bill', data=df)

# With hue
sns.swarmplot(x='day', y='total_bill', hue='sex', data=df)
```

#### **Strip Plot**
A scatter plot for categorical data.
```python
sns.stripplot(x='day', y='total_bill', data=df, jitter=True)

# Grouped strip plot
sns.stripplot(x='day', y='total_bill', hue='sex', dodge=True, data=df)
```

---

### **6. Relational Plots**

#### **Relplot**
A high-level interface for creating scatter and line plots.
```python
# Scatter plot
sns.relplot(x='total_bill', y='tip', data=df)

# Line plot
sns.relplot(x='size', y='tip', kind='line', data=df)
```

#### **Pairplot**
For visualizing relationships between multiple variables at once.
```python
# Simple pairplot
sns.pairplot(df)

# Pairplot with hue
sns.pairplot(df, hue='sex')
```

#### **Joint Plot**
For visualizing the relationship between two variables and their distributions.
```python
# Scatter plot with marginal histograms
sns.jointplot(x='total_bill', y='tip', data=df)

# Hexbin plot with density contours
sns.jointplot(x='total_bill', y='tip', kind='hex', data=df)
```

---

### **7. Heatmaps**

#### **Basic Heatmap**
For visualizing matrices (often used for correlation matrices).
```python
# Compute the correlation matrix
corr = df.corr()

# Create heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

#### **Clustermap**
Hierarchical clustering of the heatmap.
```python
sns.clustermap(corr, cmap='coolwarm', annot=True)
```

---

### **8. Facet Grids**
For plotting multiple subplots conditioned on variables.
```python
# Facet grid with scatter plot
g = sns.FacetGrid(df, col='sex')
g.map(sns.scatterplot, 'total_bill', 'tip')

# Facet grid with histograms
g = sns.FacetGrid(df, col='sex', row='smoker')
g.map(sns.histplot, 'total_bill')
```

---

### **9. Regression Plots**

#### **Linear Regression**
For visualizing linear relationships between variables.
```python
sns.lmplot(x='total_bill', y='tip', data=df)

# Regression with hue
sns.lmplot(x='total_bill', y='tip', hue='sex', data=df)
```

#### **Residual Plot**
For visualizing the difference between observed and predicted values.
```python
sns.residplot(x='total_bill', y='tip', data=df)
```

---

### **10. Customizing Plots**

#### **Titles and Labels**
```python
sns.barplot(x='day', y='total_bill', data=df)

# Add title and axis labels
plt.title('Total Bill by Day')
plt.xlabel('Day')
plt.ylabel('Total Bill')
```

#### **Changing Figure Size**
```python
plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='total_bill', data=df)
```

#### **Color Palettes**
```python
# Apply a custom color palette
sns.set_palette('Set2')

# List of available palettes: 'Set1', 'Set2', 'coolwarm', 'Blues', 'RdBu', etc.
```

---

### **11. Saving the Plot**
```python
plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='total_bill', data=df)

# Save to file
plt.savefig('plot.png', dpi=300)
```

---

### **12. Themes and Context**
Seaborn allows you to adjust the theme and context of your plots.
```python
# Set theme
sns.set_theme(style='darkgrid')

# Set context (for adjusting plot scaling)
sns.set_context('paper')  # 'notebook', 'talk', 'poster'
```

---

This **Seaborn Cheatsheet** provides you with quick access to various plot types and customizations that are frequently used in exploratory data analysis (EDA) and statistical data visualization. The power of Seaborn lies in its ability to create meaningful and beautiful visualizations with minimal code.