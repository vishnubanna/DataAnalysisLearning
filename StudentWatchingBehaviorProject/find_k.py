import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_data import readdata

def filter(df):
    """
    Filter certain columns out of a data set

    Args: 
        df (Pandas.Dataframe): The data frame that needs to be filtered

    Returns: 
        Dataframe: a new filtered data frame
    """

    problem1_df = df["vidsWatched"] >= 5
    problem1_df = df[problem1_df]
    problem1_df.head()
    xy = problem1_df.drop(['VidID', 's', 's_rel_avg', 's_tot_avg'], axis = 1)
    return xy

def find_k(plot = True):
    """
    Find the optimal K value for K means clustering

    Args: 
        Plot (bool): Do you want to distplay all plots

    Returns: 
        k (int): optimal value of k to use for k means clustering
    """

    dft, dfs = readdata("data-sets/behavior-performance.txt")
    df = filter(dfs)

    for col in df.columns[2:]:
        df[col] = (df[col] - np.mean(df[col]))/np.std(df[col])
    columns = df.columns[2:]
    print(columns)
    print(df.head)

    model = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    data = model.fit_transform(df[columns].values)

    df['tsne-2d-one'] = data[:,0]
    df['tsne-2d-two'] = data[:,1]
    df['tsne-2d-three'] = data[:,2]

    if plot:
        colr = np.linspace(0, df["userID"].values.shape[0], num = df["userID"].values.shape[0])
        ax = plt.figure(figsize=(16,7)).gca(projection='3d')
        ax.scatter(xs=df["tsne-2d-one"], ys=df["tsne-2d-two"], zs=df["tsne-2d-three"], c=colr, cmap='tab10')
        ax.set_xlabel('synthetic axis 1')
        ax.set_ylabel('synthetic axis 2')
        ax.set_zlabel('synthetic axis 3')
        ax.set_title('Reduced axis representation of data where each point represents 1 student')
        #plt.show()

    max_ks = 20
    inertias = []
    models = []
    for k in range(2, max_ks):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        inertias.append(kmeans.inertia_)
        models.append(kmeans)

    secondDeriv = [0]
    for i in range(len(inertias) - 1):
        secondDeriv.append(inertias[i+1] + inertias[i-1] - 2 * inertias[i])
    
    elbow = np.argmax(np.array(secondDeriv)) + 1
    kmeans = models[elbow + 1]
    figure = plt.figure(figsize=(6,5))
    figure.suptitle(f"plot used to find k")
    plt.plot(np.array(list(range(2, max_ks))), np.array(inertias), 'go--')
    plt.plot([elbow + 3], [inertias[elbow + 1]], 'ro')
    plt.xlabel("k")
    plt.ylabel("cluster inertia")
    #plt.show()

    if plot:
        print(df["tsne-2d-one"].shape)
        ax = plt.figure(figsize=(16,7)).gca(projection='3d')
        ax.scatter(xs=df["tsne-2d-one"], ys=df["tsne-2d-two"], zs=df["tsne-2d-three"], c=kmeans.labels_, cmap='rainbow')
        ax.scatter(xs=centers[:,0], ys=centers[:,1], zs=centers[:,2], c=np.arange(centers.shape[0]), marker='x',cmap='rainbow')
        ax.set_xlabel('synthetic axis 1')
        ax.set_ylabel('synthetic axis 2')
        ax.set_zlabel('synthetic axis 3')
        ax.set_title('Reduced axis representation of data where each point represents 1 student')
        #plt.show()
    return elbow + 3


if __name__ == "__main__":
    find_k()
    plt.show()