import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from statsmodels.graphics.mosaicplot import mosaic
import os

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

# Load the dataset
dataset = pd.read_csv("mcdonalds_data.csv")

# Check the shape of the loaded data
print("Shape of the dataset after loading:", dataset.shape)

# Filter out rows with non-numeric values in the 'VisitsPerWeek' column
non_numeric_visits = dataset[~dataset['VisitsPerWeek'].apply(lambda x: x.isdigit())]
print("Rows with non-numeric 'VisitsPerWeek' values:")
print(non_numeric_visits)

# Map categorical values in 'VisitsPerWeek' to numeric values
visits_mapping = {
    'Every day': 7,
    'Every two days': 3.5,
    'Every three days': 2.33,
    'Every four days': 1.75,
    'Every five days': 1.4,
    'Every six days': 1.17,
    'Every week': 1,
    'Every two weeks': 0.5,
    'Every three weeks': 1 / 3,
    'Every four weeks': 0.25,
    'Every month': 0.08,
    'Every two months': 0.04,
    'Every three months': 0.03,
    'Every four months': 0.02,
    'Every six months': 0.01,
    'Once a year': 1 / 12
}

dataset['VisitsPerWeek'] = dataset['VisitsPerWeek'].map(visits_mapping)

# Remove rows with NaN values in 'VisitsPerWeek'
dataset = dataset.dropna(subset=['VisitsPerWeek'])

# Encode 'Preference' column using Label Encoding
label_encoder = LabelEncoder()
dataset['Preference'] = label_encoder.fit_transform(dataset['Preference'])

# Check unique encoded values in 'Preference'
print("Encoded 'Preference' values:")
print(dataset['Preference'].unique())

# Select numeric columns for PCA
numeric_data = dataset[['Age', 'VisitsPerWeek', 'Preference']].copy()

# Check the columns selected for PCA
print("Columns in numeric_data:")
print(numeric_data.columns)

# Principal Component Analysis (PCA)
pca = PCA()

# Perform PCA if there's data available
if len(numeric_data) > 0:
    pca_result = pca.fit_transform(numeric_data)
else:
    print("No data available for PCA.")

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(numeric_data)

# Visualization
plt.figure(figsize=(9, 4))
sns.scatterplot(x="VisitsPerWeek", y="Preference", data=numeric_data, hue=kmeans.labels_, s=400, palette="Set1")
plt.title("Customer Segmentation")
plt.xlabel("Visits Per Week")
plt.ylabel("Preference")
plt.show()

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=1234)
gmm.fit(numeric_data)

# Mosaic Plot
k4_labels = kmeans.labels_
ct = pd.crosstab(k4_labels, dataset['Gender'])
plt.figure()
mosaic(ct.stack(), gap=0.01)
plt.show()

# Boxplot
df = pd.DataFrame({'Segment': k4_labels, 'Age': dataset['Age']})
plt.figure()
df.boxplot(by='Segment', column='Age')
plt.title('Age Distribution by Segment')
plt.suptitle('')
plt.show()

# Statistical Analysis
segment_info = pd.concat([pd.DataFrame({'Segment': k4_labels}),
                          dataset.groupby(k4_labels)['Age'].mean().reset_index(drop=True),
                          dataset.groupby(k4_labels)['VisitsPerWeek'].mean().reset_index(drop=True),
                          dataset.groupby(k4_labels)['Preference'].mean().reset_index(drop=True),
                          dataset.groupby(k4_labels)['Gender'].apply(lambda x: x.mode()[0]).reset_index(drop=True)],
                         axis=1)

# Print segment information
print(segment_info)

# Scatter plot for segment evaluation
plt.figure(figsize=(9, 4))
sns.scatterplot(x="VisitsPerWeek", y="Preference", data=segment_info, hue="Segment", s=400, palette="Set1")
plt.title("Segment Evaluation")
plt.xlabel("Visits Per Week")
plt.ylabel("Preference")
plt.show()
