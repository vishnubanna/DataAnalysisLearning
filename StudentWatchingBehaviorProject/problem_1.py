import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from load_data import readdata, normalize
from find_k import find_k

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
    print(problem1_df.head())
    xy = problem1_df.drop(['VidID', 's', 's_rel_avg', 's_tot_avg', 'stdPBR'], axis = 1)
    return xy

dft, dfs = readdata("data-sets/behavior-performance.txt")
xy = filter(dfs)

k = find_k(plot=False)
        
xy2 = xy.to_numpy()
xy2, m, s = normalize(xy2[:,1:])
kmeans = KMeans(n_clusters=k) #number of clusters
kmeans.fit(xy2)
centers = kmeans.cluster_centers_
figure2 = plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.05, top=0.91, hspace=.5, wspace=.5, left=.01, right=.99)
count = 1

graph_set = []
for col in range(centers.shape[-1]):
    for two in range(centers.shape[-1]):
        set_used = {xy.columns[col + 1], xy.columns[two + 1]}
        if (two != col and set_used not in graph_set):
            gr = xy.columns[col + 1] + " vs. " + xy.columns[two + 1]
            axe = figure2.add_subplot(7, 7, count, title=gr)
            axe.scatter(xy2[:,col], xy2[:,two], c=kmeans.labels_, cmap='rainbow')
            axe.set_xlabel(xy.columns[col + 1])
            axe.set_ylabel(xy.columns[two + 1])
            axe.scatter(centers[:,col], centers[:,two], c=np.arange(centers.shape[0]), marker='x', s=100, cmap='rainbow')
            count += 1
            graph_set.append(set_used)

figure2.suptitle(f"all clusters at k = {k}")
plt.show()