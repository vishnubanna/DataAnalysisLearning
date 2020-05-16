from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
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
        test_data[:,1:-1] = x_test
    else:
        x_train = train_data[:, 1:-1]
        x_test = test_data[:, 1:-1]
        m_t = None
        s_t = None
        
    y_train = train_data[:, label_cols]
    y_test = test_data[:, label_cols]
    return x_train, x_test, y_train, y_test, test_data, m_t, s_t

def mse(y,y_pred):
    """
    calculate the mean square error between y and y_pred

    Args:
        y (numpy.array): Ground truth
        y_pred (numpy.array): Prediction
    
    Returns:
        error (float): the error between y and y_pred
    """
    error = float(np.sum(np.square(y - y_pred)))/(np.shape(y)[0])
    return error

def sigmoid(t):
    """
    apply sigmoid function to exponentially scale all values in t between 1 and 0  

    Args:
        t (numpy.array): the data t apply sigmoid on 
    
    Returns:
        numpy.array: the modulated data 
    """
    return (1/(1 + np.exp(-t)))


def Poly_ridge():   
    """
    apply polynomial regression of degree 1 to the sumarised dataframe extraction from data-sets/Behavioral_Shift_S_cumulative.csv

    Args:
        None
    
    Returns:
        msev (float): the error between the validation and testing data
    """
    dft = readdata2("data-sets/Behavioral_Shift_S_cumulative.csv")
    x_yarr = filter(dft)
    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = test_train(x_yarr, norm = True)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    degree = 1
    alpha = np.logspace(start = -1, stop = 2, base = 10, num = 101)
    models = []
    mset = []
    msev = []

    for limit in alpha:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(limit))
        model.fit(X = x_train, y = y_train)
        y_trpred = model.predict(x_train)
        y_pred = model.predict(x_test)
        mset.append(mse(y_train, y_trpred))
        msev.append(mse(y_test, y_pred))
        models.append(model)
        
    model = models[np.argmin(np.array(msev))]
    print(f"\nminimum mse: {np.min(np.array(msev))}")

    l = test_full[np.argsort(test_full[:, -1]), :]
    y_line = model.predict(l[:, 1:-1])
    y_line = np.array(y_line[:])

    fig = plt.figure(figsize = (5,5))
    ind = np.linspace(0, y_line.shape[-1] - 1, num = y_line.shape[-1])
    plt.text(0.00, 0.4, 'Test MSE: %.4f' % np.min(np.array(msev)))
    plt.plot(ind, l[:, -1], label = "ground truth")
    plt.plot(ind, sigmoid(y_line), label = "prediction")
    plt.xlabel("sorted_X")
    plt.ylabel("s")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (5,5))
    sns.kdeplot(np.array(y_train), shade = True, color = 'red')
    sns.kdeplot(np.array(y_test), shade = True, color = 'blue')
    sns.kdeplot(np.array(y_line), shade = True, color = 'turquoise')
    plt.ylabel("P(x)")
    plt.xlabel("x")
    plt.show()

if __name__ == "__main__":
    Poly_ridge()