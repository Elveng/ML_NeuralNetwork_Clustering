import pandas as pd
import numpy as np
import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Reads the data from CSV file
data = pd.read_csv("part2-data.csv")

# Set the number of clusters (k) and maximum iterations
k = 3
max_iterations = 100

# Min-max normalization
data_normalized = (data - data.min()) / (data.max() - data.min())

def k_means(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    for _ in range(max_iterations):
        # Assigns each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1), axis=-1)
        
        # Updates centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Checks for convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids
def calculate_wcss(data, labels, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        centroid = centroids[i]
        wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    return wcss

def calculate_bcss(data, labels, centroids):
    bcss = 0
    overall_centroid = np.mean(data, axis=0)
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        centroid = centroids[i]
        bcss += len(cluster_points) * np.linalg.norm(centroid - overall_centroid) ** 2
    return bcss

def calculate_dunn_index(data, labels, centroids):
    min_inter_cluster_distance = math.inf
    max_intra_cluster_distance = 0
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_cluster_distance = np.linalg.norm(centroids[i] - centroids[j])
            if inter_cluster_distance < min_inter_cluster_distance:
                min_inter_cluster_distance = inter_cluster_distance
        cluster_points = data[labels == i]
        intra_cluster_distance = np.max(np.linalg.norm(cluster_points - centroids[i], axis=1))
        if intra_cluster_distance > max_intra_cluster_distance:
            max_intra_cluster_distance = intra_cluster_distance
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index

labels, centroids = k_means(data_normalized.to_numpy(), k, max_iterations=100)

# Calculating WCSS, BCSS, and Dunn Index
wcss = calculate_wcss(data_normalized.to_numpy(), labels, centroids)
bcss = calculate_bcss(data_normalized.to_numpy(), labels, centroids)
dunn_index = calculate_dunn_index(data_normalized.to_numpy(), labels, centroids)

# Recording the values in the file
with open("part2_result.txt", "w") as file:
    file.write(f"Tahsin Berk Oztekin - 21070001035\n")
    file.write(f"Used Libraries:\n")
    file.write(f"pandas: Data manipulation and analysis.\n")
    file.write(f"math: Mathematical functions.\n")
    file.write(f"tkinter: GUI framework.\n")
    file.write(f"tkinter.ttk: Themed GUI components.\n")
    file.write(f"matplotlib.pyplot: Plotting and visualization.\n\n")
    file.write(f"WCSS: {wcss}\n")
    file.write(f"BCSS: {bcss}\n")
    file.write(f"Dunn Index: {dunn_index}\n\n")

    # Printing each record with its cluster
    for i in range(len(labels)):
        file.write(f"Record {i+1} : Cluster {labels[i]+1}\n")

    # Counting the number of records in each cluster
    cluster_counts = np.bincount(labels)
    for cluster_num, count in enumerate(cluster_counts):
        file.write(f"Cluster {cluster_num+1} : {count} record\n")





# Creating a tkinter window
window = tk.Tk()
window.title("Variable Selection")
window.geometry("300x150")

# Defining the available variables
variables = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]

# Creating the dropdown menus
x_label = ttk.Label(window, text="Select variable for x-axis:")
x_label.pack()
x_variable = tk.StringVar()
x_dropdown = ttk.Combobox(window, textvariable=x_variable, values=variables)
x_dropdown.pack()

y_label = ttk.Label(window, text="Select variable for y-axis:")
y_label.pack()
y_variable = tk.StringVar()
y_dropdown = ttk.Combobox(window, textvariable=y_variable, values=variables)
y_dropdown.pack()

# Function to handle button click
def visualize():
    # Clearing the previous plot
    plt.clf()
    # Geting the selected variables
    x_var = x_variable.get()
    y_var = y_variable.get()
    
    # Visualizing the output using a scatter plot
    plt.scatter(data_normalized[x_var], data_normalized[y_var], c=labels)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title('K-means Clustering')
    plt.show()

# Creating the button
button = ttk.Button(window, text="Visualize", command=visualize)
button.pack()

# Runing the tkinter event loop
window.mainloop()
# Visualizing the output using a scatter plot
plt.scatter(data_normalized[x_variable], data_normalized[y_variable], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()





