import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Longley dataset from statsmodels datasets
data = sm.datasets.longley.load_pandas().data

# Display the first few rows of the dataset
print(data.head())
# Plot Employed against each variable
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.scatterplot(x='GNP', y='Employed', data=data, ax=axes[0, 0])
sns.scatterplot(x='Unemployed', y='Employed', data=data, ax=axes[0, 1])
sns.scatterplot(x='Armed Forces', y='Employed', data=data, ax=axes[0, 2])
sns.scatterplot(x='Population', y='Employed', data=data, ax=axes[1, 0])
sns.scatterplot(x='Year', y='Employed', data=data, ax=axes[1, 1])

# Hide the empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Calculate the correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix['Employed'].sort_values(ascending=False))

# Select the top 3 correlated variables with Employed
selected_variables = ['GNP', 'Unemployed', 'Armed Forces']

# Create regression models for "Employed" using each selected variable
X = data[selected_variables]
X = sm.add_constant(X)  # Add a constant to the model (intercept)

# Fit the model using OLS (Ordinary Least Squares)
model = sm.OLS(data['Employed'], X).fit()

# Print the model summary
print(model.summary())

# Model matrices for the champion model
X_champion = sm.add_constant(data[selected_variables])
print(X_champion.head())  # Show the first few rows of the design matrix

# Get the regression parameters (coefficients)
params = model.params
print("Regression Parameters:", params)

# Recalculate the predicted values
predicted_values = X_champion.dot(params)

# Print the first few predicted values
print("Predicted Values:", predicted_values.head())


