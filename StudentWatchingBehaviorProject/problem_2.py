#problem 2 method 1 linear ridge 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf 
import tensorflow.keras as keras
from load_data import readdata, normalize


def filter1(df):
    """
    Filter certain columns out of a data set

    Args: 
        df (Pandas.Dataframe): The data frame that needs to be filtered

    Returns: 
        Dataframe: a new filtered data frame
    """
    problem2_df = df["vidsWatched"] >= 93//2
    problem2_df = df[problem2_df]
    problem2_df.head()
    xy = problem2_df.drop(['VidID','s', 's_avg', 's_rel_avg', 'stdPBR'], axis = 1)
    print(xy.shape)
    return xy

def filter2(df):
    """
    Filter certain columns out of a data set

    Args: 
        df (Pandas.Dataframe): The data frame that needs to be filtered

    Returns: 
        Numpy.array: a numpy array representation of the filtered Dataframe
    """
    problem2_df = df["vidsWatched"] >= 5
    problem2_df = df[problem2_df]
    problem2_df.head()
    xy = problem2_df.drop(['VidID','s', 's_avg', 's_rel_avg', 'stdPBR'], axis = 1)
    print(xy.shape)
    return xy.to_numpy()

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
        train_data = np.array(x_yarray, copy = True)
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

def Method1_Regression():
    """
    apply polynomial regression of degree 2 to the sumarised dataframe extraction from data-sets/behavior-performance.txt

    Args:
        None
    
    Returns:
        None
    """
    dft, dfs = readdata("data-sets/behavior-performance.txt")
    xy = filter1(dfs)
    x_yarr = xy.to_numpy()

    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = test_train(x_yarr, norm = True)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(train_mean)

    degree = 2
    alpha = np.logspace(start = -2, stop = 3, base = 10, num = 101)

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
    y_line = model.predict(l[:,1:-1])
    fig = plt.figure(figsize = (5,5))
    ind = np.linspace(0, y_line.shape[0] - 1, num = y_line.shape[0])
    print(ind.shape, y_line.shape, y_test.shape)
    plt.scatter(ind, y_line, label = "prediction")
    plt.scatter(ind, y_test, label = "Ground Truth")
    plt.text(0.00, 0.4, 'Test mse: %.4f' % np.min(np.array(msev)))
    plt.xlabel("sorted x")
    plt.ylabel("s_avg")
    fig.suptitle("Regression predictions compared to true values in validation data")
    plt.legend()
    # plt.show()

    fig1 = plt.figure(figsize = (5,5))
    plt.semilogx(np.array(alpha), np.array(msev), label = "validation mse")
    plt.semilogx(np.array(alpha), np.array(mset), label = "training mse")
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    fig1.suptitle("Comparing Validation MSE and Training MSE relative to aplha")
    plt.legend()
    # plt.show()

    fig2 = plt.figure(figsize = (5,5))
    sns.kdeplot(y_train, shade = True, color = 'red', legend=True)
    sns.kdeplot(y_test, shade = True, color = 'blue', legend=True)
    sns.kdeplot(y_line, shade = True, color = 'turquoise', legend=True)
    fig2.suptitle("Regression prediction Distribution")
    plt.ylabel("P(x)")
    plt.xlabel("x")
    plt.show()
    pass

def get_NeuralNet(insize):
    """
    generate small MLP to use for model training and prediciton

    Args:
        insize (int): the size of the input layer
    
    Returns:
        model (TensorFlow.Keras.Sequential.model): the Neral Network Model 
    """
    model = keras.Sequential([
        keras.layers.Dense(insize, input_shape = (insize,)),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dense(8, activation = 'relu'),
        keras.layers.Dense(4, activation = 'relu'),
        keras.layers.Dense(1, activation = 'linear'),
        ])
    return model

def Method2_NeuralNet():
    """
    apply MLP Neural Net to the sumarised dataframe extraction from data-sets/behavior-performance.txt
        Network trained for:
        - epochs = 1
        - batch_size = 1
        - loss = mean squared error
        - optimizer = Adam Gradient Decent

    Args:
        None
    
    Returns:
        None
    """
    loss = 'mse'
    optimizer = keras.optimizers.Adam()
    EPOCHS = 1
    BATCH_SIZE = 1
    dft, dfs = readdata("data-sets/behavior-performance.txt")

    x_yarr = filter2(dfs)
    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = test_train(x_yarr, norm = True)

    x_train = tf.convert_to_tensor(x_train, dtype = tf.float64)
    x_test = tf.convert_to_tensor(x_test, dtype = tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype = tf.float64)
    y_test = tf.convert_to_tensor(y_test, dtype = tf.float64)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    model = get_NeuralNet(x_train.shape[-1])
    model.compile(optimizer = optimizer, loss = loss)
    model.fit(x = x_train, y = y_train, epochs = EPOCHS, verbose = 1, validation_data = (x_test, y_test), batch_size = BATCH_SIZE)
    y_trpred = model.predict(x_train)
    y_pred = model.predict(x_test)

    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(y_test, y_pred)
    print(model.summary())

    fig = plt.figure(figsize = (5,5))
    l = test_full[np.argsort(test_full[:, -1]), :]
    l2 = tf.convert_to_tensor(l[:, 1:-1], dtype = tf.float64)
    y_line = model.predict(l2)
    y_line = np.array(y_line)

    ind = np.linspace(0, y_line.shape[0] - 1, num = y_line.shape[0])
    plt.scatter(ind, y_line, label = "prediction", alpha = 0.7)
    plt.scatter(ind, y_test, label = "Ground Truth", alpha = 0.7)
    plt.text(0.5, 0.5, 'Test mse: %.4f' % m.result().numpy())
    plt.xlabel("sorted x")
    plt.ylabel("s_avg")
    fig.suptitle("Neural Net predictions compared to true values in validation data")
    plt.legend()
    # plt.show()

    fig2 = plt.figure(figsize = (5,5))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    sns.kdeplot(np.array(y_train), shade = True, color = 'red')
    sns.kdeplot(np.array(y_test), shade = True, color = 'blue')
    sns.kdeplot(np.array(y_line[:, 0]), shade = True, color = 'turquoise')
    plt.ylabel("P(x)")
    plt.xlabel("x")
    fig2.suptitle("Neural Net Prediction Probability Distrbution")
    plt.show()
    pass

if __name__ == "__main__":
    Method1_Regression()
    Method2_NeuralNet()