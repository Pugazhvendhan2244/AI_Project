```markdown
# Energy Consumption Analysis with XGBoost

In this code, we'll explore energy consumption data using the XGBoost regression model. The dataset used is "PJME_hourly.csv," and the goal is to predict energy consumption (in MW) over time.

## Data Loading and Visualization

We start by loading the dataset and visualizing the energy consumption over time using Pandas and Matplotlib.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
filepath = "C:\\Users\\prath\\Downloads\\PJME_hourly.csv"
df = pd.read_csv(filepath)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# Plot the energy consumption over time
df.plot(style='.', figsize=(15, 5), title='PJM Energy (in MW) over time')
plt.show()
```

We then split the data into training and testing sets based on a specific date.

## Feature Engineering

To improve the predictive model, we create additional time-related features, such as hour, day of the week, quarter, month, year, and more.

```python
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)
```

## Data Visualization

We visualize the energy consumption distribution by hour and month using Seaborn.

```python
import seaborn as sns

# Energy consumption distribution by hour
sns.boxplot(data=df, x='hour', y='PJME_MW')
plt.title('MW by Hour')
plt.show()

# Energy consumption distribution by month
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues')
plt.title('MW by Month')
plt.show()
```

## Model Building

We use the XGBoost regression model to predict energy consumption. The data is split into training and testing sets, and the model is trained.

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Features and target variable
features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
target = 'PJME_MW'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Build the regression model
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                       early_stopping_rounds=50, objective='reg:linear',
                       max_depth=3, learning_rate=0.01)

reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
```

## Model Evaluation

We evaluate the model's performance and visualize the feature importances.

```python
# Evaluate the model and calculate RMSE
test['prediction'] = reg.predict(X_test)
score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction'])
print(f'RMSE Score on Test set: {score:0.2f}')

# Calculate R-squared (R2) Score
from sklearn.metrics import r2_score
r2 = r2_score(test['PJME_MW'], test['prediction'])
print("R-squared (R2) Score:", r2)

# Feature Importance
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()
```

Finally, we visualize the raw data and model predictions.

```python
ax = df[['PJME_MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()
```

This code provides an example of how to analyze energy consumption data, create features, and build an XGBoost regression model for prediction.
```

