Here’s a **Matplotlib Cheatsheet** that covers the essential commands and scenarios for using Matplotlib in data visualization. It’s focused on typical tasks that data scientists handle in exploratory data analysis and visualization.

### **1. Importing Matplotlib & Setup**
```python
import matplotlib.pyplot as plt

# Inline plots for Jupyter notebooks
%matplotlib inline  

# Set plot style (optional)
plt.style.use('ggplot')  # 'seaborn', 'classic', etc.
```

### **2. Basic Plot**
```python
# Line plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plot Title')
plt.show()

# Scatter plot
plt.scatter([1, 2, 3], [4, 5, 6])

# Bar plot
plt.bar(['A', 'B', 'C'], [4, 5, 6])
```

### **3. Customizing Plot Appearance**
```python
# Set figure size
plt.figure(figsize=(10, 6))

# Set axis labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot Title')

# Change line style and color
plt.plot(x, y, color='blue', linestyle='--', linewidth=2, marker='o')

# Change font size
plt.title('Title', fontsize=20)
plt.xlabel('X-axis', fontsize=15)
plt.ylabel('Y-axis', fontsize=15)
```

### **4. Adding Grid & Legends**
```python
# Add grid
plt.grid(True)

# Add legend
plt.plot(x, y, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.legend(loc='best')  # 'upper right', 'lower left', etc.
```

### **5. Multiple Subplots**
```python
# Create subplots (2 rows, 1 column)
plt.subplot(2, 1, 1)  # First subplot
plt.plot(x, y)
plt.title('First Plot')

plt.subplot(2, 1, 2)  # Second subplot
plt.plot(x, y2)
plt.title('Second Plot')

plt.tight_layout()  # Avoid overlap between subplots
```

### **6. Scatter Plot**
```python
# Basic scatter plot
plt.scatter(x, y)

# Customizing scatter plot
plt.scatter(x, y, color='red', marker='x', s=100, alpha=0.5)

# Scatter plot with color and size
plt.scatter(x, y, c=z, s=sizes, cmap='viridis')
plt.colorbar()  # Add color bar for the plot
```

### **7. Bar Charts**
```python
# Vertical bar chart
plt.bar(categories, values)

# Horizontal bar chart
plt.barh(categories, values)

# Stacked bar chart
plt.bar(x, y1, label='Series 1')
plt.bar(x, y2, bottom=y1, label='Series 2')
plt.legend()
```

### **8. Histograms**
```python
# Basic histogram
plt.hist(data, bins=10)

# Customizing histogram
plt.hist(data, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Title')
```

### **9. Boxplot (Box-and-Whisker Plot)**
```python
# Basic boxplot
plt.boxplot(data)

# Boxplot for multiple datasets
plt.boxplot([data1, data2, data3])
plt.xticks([1, 2, 3], ['Dataset 1', 'Dataset 2', 'Dataset 3'])
```

### **10. Pie Charts**
```python
# Basic pie chart
plt.pie(sizes, labels=labels)

# Adding percentages and customizing pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
```

### **11. Error Bars**
```python
# Adding error bars
plt.errorbar(x, y, yerr=error, fmt='o', ecolor='red', capsize=5)
```

### **12. Saving Plots**
```python
# Save as PNG, PDF, etc.
plt.savefig('plot.png', dpi=300)
plt.savefig('plot.pdf')
```

### **13. Logarithmic & Other Scales**
```python
# Logarithmic scale on y-axis
plt.yscale('log')

# Set x and y axis limits
plt.xlim(0, 100)
plt.ylim(0, 500)

# Customize ticks
plt.xticks([0, 1, 2, 3, 4, 5])
plt.yticks([0, 100, 200, 300])
```

### **14. Advanced Customization**
```python
# Add annotations
plt.plot(x, y)
plt.annotate('Important point', xy=(x_value, y_value), xytext=(x_value+1, y_value+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Adding text to a plot
plt.text(x_value, y_value, 'Text on plot', fontsize=12)
```

### **15. Heatmaps**
```python
# Simple heatmap
plt.imshow(data, cmap='hot', interpolation='nearest')

# Advanced heatmap with color bar
plt.imshow(data, cmap='viridis')
plt.colorbar()
```

### **16. 3D Plots**
```python
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z)

# 3D scatter plot
ax.scatter3D(x, y, z, c=z, cmap='Greens')
```

### **17. Seaborn Integration**
```python
import seaborn as sns

# Use Seaborn styles in Matplotlib
sns.set()

# Create a Seaborn plot using Matplotlib
sns.heatmap(data, annot=True, cmap='coolwarm')
```

This cheatsheet provides a concise overview of commonly used Matplotlib features and customization options. It covers a range of use cases, from simple line plots to advanced 3D visualizations, and is perfect for quick reference when visualizing data.