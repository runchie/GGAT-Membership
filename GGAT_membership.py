import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membership Data
membership_data = {
    2561: {
        'Section': ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total'],
        'Bangkok and central': [186, 736, 3026, 8732, 12680],
        'Southern': [449, 820, 1805, 5454, 8528],
        'Northeastern': [196, 0, 11023, 22061, 33280],
        'Northern': [1505, 2488, 1014, 9387, 14394],
        'Total': [2336, 4044, 16868, 45634, 68882]
    },
    2562: {
        'Section': ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total'],
        'Bangkok and central': [74, 1723, 6327, 4689, 12813],
        'Southern': [346, 775, 3919, 1475, 6515],
        'Northeastern': [238, 379, 10125, 10149, 20891],
        'Northern': [1391, 1795, 4556, 1542, 9284],
        'Total': [2049, 4672, 24927, 17855, 49503]
    },
    2563: {
        'Section': ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total'],
        'Bangkok and central': [1216, 1038, 6105, 2653, 11012],
        'Southern': [302, 705, 3618, 743, 5368],
        'Northeastern': [318, 313, 10168, 13496, 24295],
        'Northern': [1109, 1525, 3254, 1947, 7835],
        'Total': [2945, 3581, 23145, 18839, 48510]
    },
    2564: {
        'Section': ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total'],
        'Bangkok and central': [0, 984, 4031, 480, 5495],
        'Southern': [331, 532, 1884, 992, 3739],
        'Northeastern': [0, 126, 5672, 6951, 12749],
        'Northern': [1155, 1450, 1813, 916, 5334],
        'Total': [1486, 3092, 13400, 9339, 27317]
    },
    2565: {
        'Section': ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total'],
        'Bangkok and central': [71, 357, 2247, 1505, 4180],
        'Southern': [0, 258, 2709, 310, 3277],
        'Northeastern': [744, 695, 3184, 3936, 8559],
        'Northern': [391, 0, 611, 571, 1573],
        'Total': [1206, 1310, 8751, 6322, 17589]
    }
}

# School Data
school_data = {
    2561: {
        'Region': ['Bangkok and central', 'Southern', 'Northeastern', 'Northern', 'Total'],
        'School': [36, 24, 114, 29, 203]
    },
    2562: {
        'Region': ['Bangkok and central', 'Southern', 'Northeastern', 'Northern', 'Total'],
        'School': [48, 26, 115, 37, 226]
    },
    2563: {
        'Region': ['Bangkok and central', 'Southern', 'Northeastern', 'Northern', 'Total'],
        'School': [38, 27, 110, 30, 205]
    },
    2564: {
        'Region': ['Bangkok and central', 'Southern', 'Northeastern', 'Northern', 'Total'],
        'School': [16, 15, 70, 21, 122]
    },
    2565: {
        'Region': ['Bangkok and central', 'Southern', 'Northeastern', 'Northern', 'Total'],
        'School': [15, 11, 31, 9, 66]
    }
}

# Merge Membership Data
membership_dfs = []
for year, data in membership_data.items():
    df = pd.DataFrame(data)
    df = df.set_index('Section')
    membership_dfs.append(df)

membership_merged = pd.concat(membership_dfs, keys=membership_data.keys(), names=['Year', 'Section'])

# Merge School Data
school_dfs = []
for year, data in school_data.items():
    df = pd.DataFrame(data)
    df = df.set_index('Region')
    school_dfs.append(df)

school_merged = pd.concat(school_dfs, keys=school_data.keys(), names=['Year', 'Region'])

# Reset index for better visualization
membership_merged = membership_merged.reset_index()
school_merged = school_merged.reset_index()

# Plotting Membership Data
plt.figure(figsize=(12, 6))
sns.lineplot(data=membership_merged, x='Year', y='Total', hue='Section', marker='o')

# Setting plot title and labels
plt.title('Membership Trends')
plt.xlabel('Year')
plt.ylabel('Membership Count')

# Save the plot as an image file
plt.savefig('membership_trends.png')

# Plotting School Data
plt.figure(figsize=(10, 6))
sns.barplot(data=school_merged, x='Year', y='School', hue='Region')

# Setting plot title and labels
plt.title('School Distribution')
plt.xlabel('Year')
plt.ylabel('Number of Schools')

# Save the plot as an image file
plt.savefig('school_distribution.png')

# Display the plots
plt.show()

# Clean column names
membership_merged.columns = membership_merged.columns.str.strip()
school_merged.columns = school_merged.columns.str.strip()

# Convert column types if necessary
membership_merged['Region'] = membership_merged['Region'].astype(str)
school_merged['Region'] = school_merged['Region'].astype(str)

# Merge the dataframes
correlation = membership_merged.merge(school_merged, on=['Year', 'Region']).groupby('Region').corr()
print(correlation)

#Calculate Descriptive Statistics for Membership Data:
membership_stats = membership_merged.groupby('Section').describe()

#Calculate Descriptive Statistics for School Data
school_stats = school_merged.groupby('Region').describe()

#alculate Correlation Coefficients
correlation = membership_merged.merge(school_merged, on=['Year', 'Region']).groupby('Region').corr()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# ... Code for data preprocessing and visualization ...

# Predictive Modeling: ARIMA
# Extract the total membership count for modeling
membership_total = membership_df[['Year', 'Total']]

# Set 'Year' as the index for time series analysis
membership_total = membership_total.set_index('Year')

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(membership_total, model='additive')

# Visualize the decomposed time series
fig, axes = plt.subplots(4, 1, figsize=(10, 12))
decomposition.observed.plot(ax=axes[0])
axes[0].set_ylabel('Observed')
decomposition.trend.plot(ax=axes[1])
axes[1].set_ylabel('Trend')
decomposition.seasonal.plot(ax=axes[2])
axes[2].set_ylabel('Seasonal')
decomposition.resid.plot(ax=axes[3])
axes[3].set_ylabel('Residual')
plt.suptitle('Decomposition of Membership Time Series')
plt.tight_layout()
plt.show()

# Perform Time Series Forecasting using ARIMA
model = ARIMA(membership_total, order=(1, 1, 1))
model_fit = model.fit()

# Predict future values
future_years = range(2566, 2571)  # Assuming 5 future years
forecast = model_fit.forecast(steps=len(future_years))

# Visualize the forecasted values
plt.plot(membership_total.index, membership_total, label='Actual')
plt.plot(future_years, forecast, label='Forecast')
plt.title('Membership Time Series Forecast')
plt.xlabel('Year')
plt.ylabel('Membership Count')
plt.legend()
plt.show()

# Update column names in the membership data
membership_data[2561]['Section'] = ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total']
membership_data[2562]['Section'] = ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total']
membership_data[2563]['Section'] = ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total']
membership_data[2564]['Section'] = ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total']
membership_data[2565]['Section'] = ['Little bird', 'Blue Birds', 'Guides', 'Senior Guides', 'Total']

# Merge membership and school data
membership_merged = pd.concat([pd.DataFrame(membership_data[year]) for year in membership_data], ignore_index=True)
school_merged = pd.concat([pd.DataFrame(school_data[year]) for year in school_data], ignore_index=True)

# Update column names in the membership_merged DataFrame
membership_merged.columns = ['Year', 'Region'] + membership_merged.columns.tolist()[2:]

# Convert 'Region' column to string type
membership_merged['Region'] = membership_merged['Region'].astype(str)

# Perform correlation analysis
correlation = membership_merged.merge(school_merged, on=['Year', 'Region']).groupby('Region').corr()
