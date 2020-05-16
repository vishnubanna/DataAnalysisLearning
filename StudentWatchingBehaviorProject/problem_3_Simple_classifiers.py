from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import svm
import tensorflow as tf 
import tensorflow.keras as keras
from scipy.stats import mode
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import readdata2, normalize

def filter(df):
    """
    Filter certain columns out of a data set and prepare the data for classification.
        shuffle the data reduce size such that both classes have the same number of data points

    Args: 
        df (Pandas.Dataframe): The data frame that needs to be filtered

    Returns: 
        Numpy.array: a numpy array representation of the filtered Dataframe
    """
    df = df.sort_values("s")
    #xy = df.drop(['VidID', 'stdPBR', 'cumulative_vids', 'cumulative_s','cumulative_avg'], axis = 1)
    xy = df.drop(['VidID', 'stdPBR'], axis = 1)
    xy = xy.to_numpy()
    
    # balance the number of 0 values and 1 values
    y_1 = np.argwhere(xy[:, -1] == 1).flatten()
    y_0 = np.argwhere(xy[:, -1] == 0).flatten()
    reduct = min(y_1.size, y_0.size)
    np.random.shuffle(y_1)
    np.random.shuffle(y_0)
    y_1 = y_1[0: reduct]
    y_0 = y_0[0: reduct]
    yidx = np.concatenate([y_0, y_1], axis = 0).flatten()
    np.random.shuffle(yidx)
    xy = xy[yidx, :]
    print(mode(xy[:, -1]), np.median(xy[:, -1]))
    return xy

def filter_knn(df):
    """
    Filter certain columns out of a data set and prepare the data for classification.
        shuffle the data reduce size such that both classes have the same number of data points

    Args: 
        df (Pandas.Dataframe): The data frame that needs to be filtered

    Returns: 
        Numpy.array: a numpy array representation of the filtered Dataframe
    """
    df = df.sort_values("s")
    xy = df.drop([ 'fracPlayed', 'stdPBR', 'numRWs', 'cumulative_vids', 'cumulative_s','cumulative_avg', 'shift_s'], axis = 1)
    xy = xy.to_numpy()
    
    # balance the number of 0 values and 1 values
    y_1 = np.argwhere(xy[:, -1] == 1).flatten()
    y_0 = np.argwhere(xy[:, -1] == 0).flatten()
    reduct = min(y_1.size, y_0.size)
    np.random.shuffle(y_1)
    np.random.shuffle(y_0)
    y_1 = y_1[0: reduct]
    y_0 = y_0[0: reduct]
    yidx = np.concatenate([y_0, y_1], axis = 0).flatten()
    np.random.shuffle(yidx)
    xy = xy[yidx, :]
    print(mode(xy[:, -1]), np.median(xy[:, -1]))
    return xy

# need to balnce so there are the same number of 0 points as 1 points
def test_train(x_yarr, offset = 0, x_split = 0.9, nshuffles = 1, x_cols = 1, label_cols = -1, norm = True):
    """
    split the data into train and test data

    Args: 
        x_yarr (numpy.array): An array contianing the data to seperate
        offset (int): the offset of the data to use as test, if 0, the end of the array is used
        x_split (float): percent of th data to use as training 
        nshuffles (int): how many times to shuffle data
        x_cols (int): the colomn axis starting point of data in x_yarr
        label_cols (int): the colomn axis starting point of labels in x_yarr
        norm (bool): normalize the data if True

    Returns: 
        x_train (numpy.array): x colomns for training 
        x_test (numpy.array): x colomns for testing
        y_train (numpy.array): y colomns for training
        y_test (numpy.array): y colomns for testing
        test_data (numpy.array): all the columns of test_data (both x_test and y_test)
        m_t (numpy.array): mean of the columns
        s_t (numpy.array): std of the columns
    """
    for i in range(nshuffles):
        np.random.shuffle(x_yarr)

    if offset == 0:
        train_size = int(x_yarr.shape[0] * x_split)
        train_data = x_yarr[:train_size]
        test_data = x_yarr[train_size:]
    else:
        test_size = int(x_yarr.shape[0] * (1 - x_split))
        test_data = x_yarr[offset * test_size:(offset + 1) * test_size]
        train_data = np.array(x_yarr, copy = True)
        train_data = np.delete(train_data, slice(offset * test_size, (offset + 1) * test_size), axis = 0)
    
    if norm:
        x_train, m_t, s_t = normalize(train_data[:, 1:-1])
        x_test, m_t , s_t = normalize(test_data[:,1:-1], m_t, s_t)
    else:
        x_train = train_data[:, 1:-1]
        x_test = test_data[:, 1:-1]
        m_t = None
        s_t = None
        
    y_train = train_data[:, label_cols]
    y_test = test_data[:, label_cols]
    return x_train, x_test, y_train, y_test, test_data, m_t, s_t


def getdata(filter_func = filter):
    """
    load and filter data then find train test split

    Args:
        filter_func (python function): function to use to filter the data 

    Returns: 
        x_train (numpy.array): x colomns for training 
        x_test (numpy.array): x colomns for testing
        y_train (numpy.array): y colomns for training
        y_test (numpy.array): y colomns for testing
        test_full (numpy.array): all the columns of test_data (both x_test and y_test)
        train_mean (numpy.array): mean of the columns
        train_std (numpy.array): std of the columns
    """
    dft = readdata2("data-sets/Behavioral_Shift_S_cumulative.csv")
    x_yarr = filter_func(dft)
    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = test_train(x_yarr, norm = False)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test, test_full, train_mean, train_std

def plot(model, test_full, x_train, x_test, y_train, y_test, y_line, l, metric, title):
    """
    generate plots to detemine how will the modle was able to classify the data

    Args:
        model (sklearn.model): the trained model used to make predictions on y_test
        x_train (numpy.array): x colomns for training 
        x_test (numpy.array): x colomns for testing
        y_train (numpy.array): y colomns for training
        y_test (numpy.array): y colomns for testing
        y_line (numpy.array): the prediction of the model
        test_full (numpy.array): all the columns of test_data (both x_test and y_test)
        metric (float): the metric used to judge the model
        title (str): the name of the classifier

    Returns: 
        None
    """
    fig = plt.figure(figsize = (5,5))
    ind = np.linspace(0, y_line.shape[-1] - 1, num = y_line.shape[-1])
    plt.text(0.00, 0.4, 'Test Accuracy: %.4f' % np.abs(metric))
    plt.plot(ind, l[:, -1], label = "ground truth", color = 'red')
    plt.scatter(ind, y_line, label = "prediction", alpha=0.1)
    plt.xlabel("sorted_X")
    plt.ylabel("s")
    plt.title(title)
    plt.legend()
    # plt.show()

    fig = plt.figure(figsize = (5,5))
    sns.kdeplot(np.array(y_train), shade = True, color = 'red', legend = True)
    sns.kdeplot(np.array(y_test), shade = True, color = 'blue')
    sns.kdeplot(np.array(y_line), shade = True, color = 'turquoise')
    plt.ylabel("P(x)")
    plt.xlabel("x")
    plt.title(f"{title} Prediction Distribution")
    plt.show()
    pass 


def model_fit_sklearn(model, filter_func = filter, title = None):
    """
    a template function to train most sklearn models on data-sets/Behavioral_Shift_S_cumulative.csv

    Args: 
        model (sklearn.model): the initialized untrained model to fit the training data too
        filter (python function): the function used to filter data 
        title (str): the name of the model being used 

    Returns:
        mset (float): the training accuracy 
        msev (float): the testing accuracy 
    """
    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = getdata(filter_func = filter_func)

    model.fit(X = x_train, y = y_train)
    y_trpred = model.predict(x_train)
    y_pred = model.predict(x_test)
    mset = roc_auc_score(y_train, y_trpred)
    msev = roc_auc_score(y_test, y_pred)

    l = test_full[np.argsort(test_full[:, -1]), :]
    y_line = model.predict(l[:, 1:-1])
    y_line = np.array(y_line[:])
        
    print(f"\nMaximum Accuracy: {msev}")
    plot(model, test_full, x_train, x_test, y_train, y_test, y_line, l, msev, title)
    return mset, msev

def Logistic_Classifier():
    """
    apply Logistic regression to dataset 

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    print(f"Logistic regression")
    model = LogisticRegression(C=1e5, penalty = 'l2', solver = 'lbfgs', max_iter = 1000)
    mset, msev = model_fit_sklearn(model, title = "Logistic regression")
    return msev

def SVM_Classifier():
    """
    apply SVM Classifier to dataset 

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    print("SVM Classifier")
    model = svm.SVC(kernel='linear',gamma='auto')
    mset, msev = model_fit_sklearn(model, title = "SVM Classifier")
    return msev

def GaussianBayes_Classifier():
    """
    apply Gaussian Bayes Classifier to dataset 

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    print("Gaussian Bayes")
    model = GaussianNB()
    mset, msev = model_fit_sklearn(model, title = "Gaussian Bayes")
    return msev

def RandomForrest_Classifier():
    """
    apply Random Forrest Classifier to dataset 

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    print("Random Forrest")
    model = RandomForestClassifier(n_estimators = 100)
    mset, msev = model_fit_sklearn(model, title = "Random Forrest")
    return msev

def Knn_Classifier():
    """
    apply kNN Classifier to data-sets/Behavioral_Shift_S_cumulative.csv

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = getdata(filter_func = filter_knn)

    models = []
    mset = []
    msev = []

    for i in range(2, 10, 1):
        model = KNeighborsClassifier(n_neighbors = i)
        model.fit(X = x_train, y = y_train)
        y_trpred = model.predict(x_train)
        y_pred = model.predict(x_test)
        mset.append(accuracy_score(y_train, y_trpred))
        msev.append(accuracy_score(y_test, y_pred))
        models.append(model)

    model = models[np.argmax(np.array(msev))]
    print(f"\nMaximum Accuracy: {np.max(np.array(msev))}")

    l = test_full[np.argsort(test_full[:, -1]), :]
    y_line = model.predict(l[:, 1:-1])
    y_line = np.array(y_line[:])

    plot(model, test_full, x_train, x_test, y_train, y_test, y_line, l,  msev[np.argmax(np.array(msev))], title = "kNN classifier")
    return msev

if __name__ == "__main__":
    Logistic_Classifier()
    GaussianBayes_Classifier()
    Knn_Classifier()
    #RandomForrest_Classifier()

    # the models bellow will take a while to run 
    # so uncomment them and compile them if you absolutlye need too
    # SVM_Classifier()
    
