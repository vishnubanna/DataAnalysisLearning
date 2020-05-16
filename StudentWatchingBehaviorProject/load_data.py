import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import math

def readdata(filename):
    """
    Read csv file with tab delimiters

    Args: 
        filename (str): the file you want to read 
    
    Returns: 
        data (pandas.Dataframe): the data in filename
        sumarised (pandas.Dataframe): the data in which all data points are sumarised for each student.
    """
    df = pd.read_csv(filename, delimiter = "\t")
    data, sumarised = get_arranged(df)
    users = list(sumarised.userID)
    tdp = list(data.userID)
    print(f"total data points: {len(tdp)}, total unique students: {len(users)}\n")
    return data, sumarised

def readdata2(filename):
    """
    Read comma delimited csv files that were generated from original data set

    Args: 
        filename (str): the file you want to read 
    
    Returns: 
        df (pandas.Dataframe): the data in filename
    """
    df = pd.read_csv(filename, delimiter = ",")
    df["cumulative_avg"] = df["cumulative_avg"].apply(lambda x: 0 if np.isnan(x) else x)
    df["s_"] = df["s"]
    df.drop("s", inplace = True, axis = 1)
    df.drop("Unnamed: 0", inplace = True, axis = 1)
    df["s"] = df["s_"]
    df.drop("s_", inplace = True, axis = 1)
    df = df.fillna(0)
    ldf = list(df.userID)
    print(f"total data points: {len(ldf)}\n")
    return df

def readdata_v(filename):
    """
    Read csv file with tab delimiters

    Args: 
        filename (str): the file you want to read 
    
    Returns: 
        data (pandas.Dataframe): the data in file
        sumarised (pandas.Dataframe): data sumarised on a per video basis
    """
    df = pd.read_csv(filename, delimiter = "\t")
    data, sumarised = get_VidID(df)
    users = list(sumarised.VidID)
    tdp = list(data.VidID)
    print(f"total data points: {len(tdp)}, total unique: {len(users)}\n")
    return data, sumarised

def normalize(data, m = np.array([None]), s = np.array([None])):
    """
    normalize a numy array for mean = 0 and standard deviation = 1

    Args: 
        data (numpy.array): data to nomalize
        m (numpy.array): the mean to normalize around if none is provided it is calculated
        s (numpy.array): the std to normalize around if none is provided it is calculated
    
    Returns: 
        data (numpy.array): normalized data
        m_t (numpy.array): the mean of the data
        s_t (numpy.array): the std of the data
    """
    if np.any(m == None):
        m = []
        for col in range(data.shape[-1]):
            m.append(np.mean(data[:, col]))
        m = np.array(m)
    if np.any(s == None):
        s = []
        for col in range(data.shape[-1]):
            s.append(np.std(data[:, col]))
        s = np.array(s)
    return (data - m)/s, m, s

def get_arranged(df):
    """
    sumarize a dataframe so column 0 is grouped an sorted

    Args: 
        df (pandas.Dataframe): data frame to sumarise 
    
    Returns: 
        ndf (pandas.Dataframe): sorted dataframe
        colpdf (pandas.Dataframe): sumarised datafram
    """
    ndf = df.sort_values("userID")
    cols = dict()
    for col in ndf.columns[1:]:
        if col == "VidID":
            cols[col] = list
        elif col == 's':
            cols[col] = np.sum
        else:
            cols[col] = np.mean
    maxvids = max(df.VidID) + 1
    colapdf = df.groupby("userID").agg(cols).reset_index()
    colapdf['vidsWatched'] = colapdf["VidID"].str.len()
    colapdf['vids_gt_5'] = colapdf["vidsWatched"].apply(lambda x: x if x >= 5 else math.inf)
    colapdf['s_avg'] = colapdf["s"].apply(lambda x: x/maxvids)
    colapdf['s_rel_avg'] = colapdf["s"]/(colapdf["vids_gt_5"]) # given you watched n videos, what was your avg on the vids you watched
    colapdf['s_tot_avg'] = colapdf["s"]/maxvids
    colapdf.drop("vids_gt_5", inplace = True, axis = 1)
    return ndf, colapdf

def modulate(df):
    """
    calulate cumulated s, vids, and shifted_s for each student

    Args: 
        df (pandas.Dataframe): data frame to modulate
    
    Returns: 
        findf (pandas.Dataframe): to 
    """
    keys = list(set(list(df.userID)))
    findf = []
    for key in keys:
        mask = df.groupby('userID')['userID'].transform(lambda x: x == key)
        dfn = df[mask].sort_values("VidID")
        dfn["cumulative_vids"] = 1
        dfn["cumulative_vids"] = dfn["cumulative_vids"].cumsum()
        dfn["cumulative_vids"] = dfn["cumulative_vids"] - 1
        dfn["shift_s"] = dfn['s'].shift(1).fillna(0)
        dfn["cumulative_s"] = dfn['s'].shift(1)
        dfn["cumulative_s"] = dfn["cumulative_s"].fillna(0)
        dfn["cumulative_s"] = dfn['s'].cumsum()
        dfn["cumulative_s"] = dfn["cumulative_s"] - dfn['s']
        dfn["cumulative_avg"] = dfn["cumulative_s"]/dfn["cumulative_vids"]
        findf.append(dfn) 
    findf = pd.concat(findf)
    xv = findf.to_numpy()
    print (xv.shape)
    return findf


def get_VidID(df):
    """
    sumarize a dataframe so column VidID is grouped an sorted

    Args: 
        df (pandas.Dataframe): data frame to sumarise 
    
    Returns: 
        ndf (pandas.Dataframe): sorted dataframe
        colpdf (pandas.Dataframe): sumarised datafram
    """
    ndf = df[df.duplicated("VidID", keep=False)].sort_values("VidID")
    cols = dict()
    for col in ndf.columns[:]:
        if col != 'VidID':
            if col == "userID":
                cols[col] = list
            else:
                cols[col] = np.sum
    colapdf = df[df.duplicated("VidID", keep=False)].groupby("VidID").agg(cols).reset_index()
    colapdf['studentsWatched'] = colapdf["userID"].str.len()
    colapdf.drop("userID", inplace = True, axis = 1)
    colapdf['s_avg'] = colapdf["s"]/colapdf['studentsWatched']
    for col in colapdf.columns[:]:
        if col != 'VidID' and col != 's_avg' and col != 's' and col != 'studentsWatched':
            colapdf[col] = colapdf[col]/colapdf['studentsWatched']
    return ndf, colapdf

def plot_df(data, filename = None, normal = False, rs = None, cs = None, kernel=['gau'], kde = True, histdist = True, relplots = True, pp = False):
    """
    plot a data frame in various ways and save data frame to csv

    Args: 
        filename (str): file to save your dataframe too
        noraml  (bool): normalize before plotting if True
        rs (int): rows in a of subplot graph
        cs (int): cols in a of subplot graph
        kernel (list:str): kernels to use in kde
        histdata (bool): plot histogram and kde
        relplots (bool): plot column "s" relative to other colomns
        pp (bool): plot a pairplot for all the data frames 
    
    """
    if filename != None:
        save = data.set_index(data.columns[0])
        save.to_csv(filename, sep = "\t")

    if rs != None and cs != None:
        fig = plt.figure()
        pallete = itertools.cycle(sns.color_palette())
        j = 0
    if histdist:
        shade = False
    else:
        shade = True
    for i, key in enumerate(data.columns[2:]):
        if normal:
            dp = normalize(data[key])
        else:
            dp = data[key]
        if (rs == None and cs == None):
            if kde:
                sns.kdeplot(dp, shade = shade)
            if histdist:
                sns.distplot(dp)
        else:
            axes = fig.add_subplot(rs, cs, i + 1)
            c = next(pallete)
            if kde:
                for kern in kernel:
                    sns.kdeplot(dp, shade = shade, ax = axes, kernel=kern, color = c)
            if histdist:
                sns.distplot(dp, ax = axes, color = c)
            j = i
    if (rs != None and cs != None and relplots):
        j += 1
        c = next(pallete)
        try:
            with sns.axes_style('white'):
                plot_key = 's_rel_avg'
                for i, key in enumerate(data.columns[2:]):
                    if key != plot_key:
                        sns.jointplot(plot_key, key, data, kind = 'reg', color = next(pallete))
        except:
            with sns.axes_style('white'):
                sns.jointplot('s', 'fracPlayed', data, kind = 'reg', color = next(pallete)) # fraction of videos competed
                sns.jointplot('fracComp', 'fracSpent', data, kind = 'reg', color = next(pallete)) # undefined relation
                sns.jointplot('s', 'studentsWatched', data, kind = 'reg', color = next(pallete))
    if pp:
        temp = data.drop(data.columns[0:1], axis = 1)
        sns.pairplot(temp, kind = 'reg' , diag_kind = 'kde', height=10)
    plt.show()
    pass

def dist_data(data, rs = None, cs = None, k = 0):
    """
    estimate the distirbution of each colomn

    Args: 
        data (pandas.Dataframe): dataframe holding data
        rs (int): rows in a of subplot graph
        cs (int): cols in a of subplot graph
        k  (int): which distirbution to try
    """
    fig = plt.figure()
    pallete = itertools.cycle(sns.color_palette())
    j = 0
    dists = ['norm', 'cauchy', 'cosine', 'expon', 'uniform', 'laplace', 'wald', 'rayleigh']
    for i, key in enumerate(data.columns[2:]):
        axes = fig.add_subplot(rs, cs, i + 1)
        c = next(pallete)
        datavals = stats.probplot(data[key], plot=axes, dist = dists[k])
    plt.show()
    pass

def norm_df(df):
    """
    normalize a data frame 

    Args: 
        df (pandas.Dataframe): dataframe holding data
    
    Returns:
        mean (float): mean of each column
        stds (float): std of each column
        df (pandas.Dataframe): normalized dataframe
    """
    means = []
    stds = []
    for i, key in enumerate(df.columns[2:]):
        means.append(np.mean(df[key]))
        stds.append(np.std(df[key]))
        df[key] = df[key].apply(lambda x : (x - np.mean(df[key]))/np.std(df[key]))
    return means, stds, df
