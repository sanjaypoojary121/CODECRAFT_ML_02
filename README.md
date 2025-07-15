CODECRAFT_ML_02: Customer Segmentation using K-Means Clustering

Objective:

This project is part of my Machine Learning internship at CODE CRAFT.

The goal is to perform "customer segmentation" using K-Means clustering to group retail customers based on their purchase behavior. This helps businesses:

- Identify customer groups with similar traits
- Design targeted marketing strategies
- Improve customer satisfaction and retention

Dataset:

The dataset is from the [Kaggle - Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

Features Used:
- Annual Income (k$) — Annual income of the customer
- Spending Score (1-100) — Score assigned based on customer spending behavior

Other features present but not used in clustering:
- CustomerID
- Gender
- Age

Steps Performed:

> Loaded dataset from CSV file  
> Selected relevant numeric features  
> Scaled data using StandardScaler  
> Used Elbow Method to find the optimal number of clusters  
> Applied K-Means clustering  
> Visualized customer segments using scatter plots  
> Saved clustered data to CSV for further analysis

How to Run:

1. Clone this repository:
    git clone https://github.com/your-username/CODECRAFT_ML_02.git
    

2. Place "Mall_Customers.csv" in the same directory as the Python script.

3. Install required Python packages:
    pip install pandas numpy matplotlib seaborn scikit-learn

4. Run the Python script:
    python customer_segmentation.py

 Output:

- "Elbow curve" to determine the optimal number of clusters
- "Scatter plot" visualizing customer clusters
- Cluster labels added to each customer in the output CSV:
    customers_with_clusters.csv

Example cluster-labeled data:
CustomerID Gender Age Annual Income (k$) Spending Score (1-100) Cluster
0 1 Male 19 15 39 4
1 2 Male 21 15 81 2
2 3 Female 20 16 6 1
3 4 Female 23 16 77 2
4 5 Female 31 17 40 4

Author:

- Name: Sanjaya S.
- Role: Machine Learning Intern at CODE CRAFT
