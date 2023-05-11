# %% [markdown]
# # Assignment 3 : Clustering and Fitting 

# %% [markdown]
# Dataset Description : https://data.worldbank.org/topic/climate-change 
# 
# Download Link : https://api.worldbank.org/v2/en/topic/19?downloadformat=csv
# 
# Indicator : https://data.worldbank.org/indicator

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.optimize import curve_fit

warnings.simplefilter(action='ignore', category=FutureWarning)



# %%
# Load the four indicator datasets
CO2Emission_DF= pd.read_excel("Dataset/CO2_EMISSIONS.xls")
ForestArea_DF = pd.read_excel("Dataset/FOREST_AREA.xls")
PopulationGrowth_DF = pd.read_excel("Dataset/POPULATION_GROWTH.xls")

# %%
def explore_dataset(dataset):
    """
    This function takes a dataset as input and returns the number of rows and columns, as well as the names of the columns (features/variables) in the dataset.
    """
    data = pd.read_excel(dataset)
    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns')
    print(f'The columns in the dataset are: {list(data.columns)}')
    print(len(data))


# %%
explore_dataset("Dataset/CO2_EMISSIONS.xls")

# %% [markdown]
# # Carbondioxide Emission 

# %%
# Load the four indicator datasets
CO2Emission_DF = pd.read_excel("Dataset/CO2_EMISSIONS.xls")
ForestArea_DF = pd.read_excel("Dataset/FOREST_AREA.xls")
PopulationGrowth_DF = pd.read_excel("Dataset/POPULATION_GROWTH.xls")

def explore_dataset(dataset):
    """
    This function takes a dataset as input and returns the number of rows and columns, as well as the names of the columns (features/variables) in the dataset.
    """
    data = pd.read_excel(dataset)
    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns')
    print(f'The columns in the dataset are: {list(data.columns)}')
    print(len(data))

def cluster_and_plot(df, feature_name):
    """
    Perform clustering analysis and plot the clusters.
    :param df: DataFrame containing the data
    :param feature_name: Name of the feature to be plotted on the y-axis
    """
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Choose features for clustering
    X = df.drop(['Country Code'], axis=1)
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Determine optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for '+feature_name)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    # Fit KMeans with optimal number of clusters
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    # Add cluster labels to dataframe
    df['Cluster'] = kmeans.labels_
    # Plot clusters and centers
    for cluster_num in range(3):
        plt.scatter(X.columns, kmeans.cluster_centers_[cluster_num], label=f'Cluster {cluster_num}')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('Year')
    plt.ylabel(feature_name)
    plt.show()

def fit_polynomial(df):
    """
    Fit a polynomial curve to the data and plot the curve along with the confidence range.
    :param df: DataFrame containing the data
    """
    # Load dataframe
    dataframe = pd.read_excel(df)
    # Store country codes in a separate variable
    country_codes = dataframe['Country Code']
    # Clean dataframe
    dataframe.dropna(inplace=True)
    # Remove country codes from dataframe
    dataframe.drop('Country Code', axis=1, inplace=True)
    # Normalize dataframe
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(dataframe)
    # Fit KMeans model to dataframe
    kmeans  = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_norm)
    # Get cluster labels
    labels = kmeans.labels_
    # Perform PCA
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_norm)
    # Store country codes for remaining rows in a separate variable
    country_codes_cleaned = country_codes[dataframe.index]
    # Add country codes and cluster labels to data_plot dataframe
    data_plot = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'cluster': labels, 'country': country_codes_cleaned})
    # Visualize clusters
    sns.scatterplot(data=data_plot, x='x', y='y', hue='cluster', palette='Set1')
    plt.title('Clusters')
    # Add legend
    legend_labels = ['Cluster ' + str(i+1) for i in range(len(set(kmeans.labels_)))]
    plt.legend(title='Cluster', labels=legend_labels, loc='lower left')
    plt.show()


def fit_curve(df):
    """
    Fit a polynomial curve to the data and plot the curve along with the confidence range.
    :param df: DataFrame containing the data
    """
    # Load data for the United States
    data = pd.read_excel(df)
    us_data = data.loc[data['Country Code'] == 'USA']
    
    # Define function to fit
    def poly_func(x, a, b, c):
        return a + b*x + c*x**2
    
    # Fit function to data
    x = np.array(range(1990, 2020))
    y = np.array(us_data.iloc[0, 1:])
    popt, pcov = curve_fit(poly_func, x, y)
    
    # Generate predictions for future years
    future_x = np.array(range(2020, 2040))
    future_y = poly_func(future_x, *popt)
    
    # Estimate lower and upper limits of confidence range
    perr = np.sqrt(np.diag(pcov))
    lower_bound = future_y - perr[0]*2
    upper_bound = future_y + perr[0]*2
    
    # Plot best fitting function and confidence range
    plt.plot(x, y, 'o', label='Data')
    plt.plot(future_x, future_y, label='Model')
    plt.fill_between(future_x, lower_bound, upper_bound, alpha=0.2, label='Confidence Range')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (kt)')
    plt.title('CO2 Emissions in the United States')
    plt.legend()
    plt.show()

# Call the functions
explore_dataset("Dataset/CO2_EMISSIONS.xls")
cluster_and_plot(CO2Emission_DF, 'Carbon Emissions (metric tons per capita)')
fit_curve("Dataset/CO2_EMISSIONS.xls")
cluster_and_plot(ForestArea_DF, 'Forest Area')
cluster_and_plot(PopulationGrowth_DF, 'Population Growth')

# Load dataframe
dataframe = pd.read_excel("Dataset/CO2_EMISSIONS.xls")

# Store country codes in a separate variable
country_codes = dataframe['Country Code']

# Clean dataframe
dataframe.dropna(inplace=True)

# Remove country codes from dataframe
dataframe.drop('Country Code', axis=1, inplace=True)

# Normalize dataframe
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(dataframe)

# Fit KMeans model to dataframe
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(data_norm)

# Get cluster labels
labels = kmeans.labels_

# Perform PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_norm)

# Store country codes for remaining rows in a separate variable
country_codes_cleaned = country_codes[dataframe.index]

# Add country codes and cluster labels to data_plot dataframe
data_plot = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'cluster': labels, 'country': country_codes_cleaned})

# Visualize clusters
sns.scatterplot(data=data_plot, x='x', y='y', hue='cluster', palette='Set1')
plt.title('Clusters')

# Add legend
legend_labels = ['Cluster ' + str(i + 1) for i in range(len(set(kmeans.labels_)))]
plt.legend(title='Cluster', labels=legend_labels, loc='lower left')

# Add country labels
for i, txt in enumerate(data_plot['country']):
    try:
        plt.annotate(txt, (data_plot['x'][i], data_plot['y'][i]))
    except:
        pass

plt.show()


# Load _dataframe
_dataframe = pd.read_excel("Dataset/POPULATION_GROWTH.xls")

# Store country codes in a separate variable
country_codes = _dataframe['Country Code']

# Clean _dataframe
_dataframe.dropna(inplace=True)

# Remove country codes from _dataframe
_dataframe.drop('Country Code', axis=1, inplace=True)

# Normalize _dataframe
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(_dataframe)

# Fit KMeans model to _dataframe
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(data_norm)

# Get cluster labels
labels = kmeans.labels_

# Perform PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_norm)

# Store country codes for remaining rows in a separate variable
country_codes_cleaned = country_codes[_dataframe.index]

# Add country codes and cluster labels to data_plot dataframe
data_plot = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'cluster': labels, 'country': country_codes_cleaned})

# Visualize clusters
sns.scatterplot(data=data_plot, x='x', y='y', hue='cluster', palette='Set1')
plt.title('Clusters')

# Add legend
legend_labels = ['Cluster ' + str(i+1) for i in range(len(set(kmeans.labels_)))]
plt.legend(title='Cluster', labels=legend_labels, loc='lower left')


plt.show()