
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

print("First five rows of the dataset:")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected features for clustering:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data example:")
print(X_scaled[:5])

sse = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()

k = 5

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

clusters = kmeans.labels_

df['Cluster'] = clusters

print("\nCluster labels added to dataframe:")
print(df.head())

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set1',
    s=100
)
plt.title(f'Customer Segmentation (k={k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

df.to_csv("customers_with_clusters.csv", index=False)
print("\nClustered data saved to customers_with_clusters.csv")