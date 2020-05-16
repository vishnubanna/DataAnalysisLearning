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
    xy = df.drop(['stdPBR', 'cumulative_vids'], axis = 1)
    #print(xy.head(5))
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
    #print(mode(xy[:, -1]), np.median(xy[:, -1]))
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

def one_hot_encode(arr):
    """
    one hot encode the data set in preperation for classification 

    Args:
        arr (numpy.array): array to one hot encode
    
    Returns:
        numpy.array: one hot encoded matrix
    """
    a = []
    for value in arr:
        if value == 0:
            a.append(1)
        else:
            a.append(0)
    a = np.array(a)
    
    arr = np.expand_dims(arr, axis = 1)
    a = np.expand_dims(a, axis = 1)
    
    arr = np.concatenate([a, arr], axis = -1)
    return np.array(arr)

def NeuralNet_Classifier():
    """
    apply Simple MLP NN Classifier to data-sets/Behavioral_Shift_S_cumulative.csv
        Network trained for:
        - epochs = 10
        - batch_size = 5
        - loss = CategoricalCrossentropy
        - optimizer = Adam Gradient Decent

    Args: 
        None
    
    Returns:
        msev (float): the accuracy of the model 
    """
    dft = readdata2("data-sets/Behavioral_Shift_S_cumulative.csv")
    x_yarr = filter(dft)

    x_train, x_test, y_train, y_test, test_full, train_mean, train_std = test_train(x_yarr, norm = True)
    y_train = y_train.astype('int')
    y_train = one_hot_encode(y_train)
    y_test = y_test.astype('int')
    y_test = one_hot_encode(y_test)

    x_train = tf.convert_to_tensor(x_train, dtype = tf.float64)
    x_test = tf.convert_to_tensor(x_test, dtype = tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype = tf.float64)
    y_test = tf.convert_to_tensor(y_test, dtype = tf.float64)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train[0:5])


    model = keras.Sequential([
        keras.layers.Dense(x_train.shape[-1], input_shape = (x_train.shape[-1],)),
        keras.layers.Dense(512, activation = 'relu'), 
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(2, activation = 'softmax'),
    ])

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer = keras.optimizers.Adam(), loss = loss, metrics = [keras.metrics.CategoricalAccuracy()])
    model.fit(x = x_train, y = y_train, epochs = 10, verbose = 1, validation_data = (x_test, y_test), batch_size = 5)
    y_trpred = model.predict(x_train)
    y_pred = model.predict(x_test)

    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(y_test, y_pred)
    print(model.summary())

    l = test_full[np.argsort(test_full[:, -1]), :]
    l2 = tf.convert_to_tensor(l[:, 1:-1], dtype = tf.float64)
    y_line = model.predict(l2)
    y_line = np.array(y_line)
    y_line = np.argmax(y_line, axis = 1).flatten()
    y_line[y_line == 0] = 2
    y_line[y_line == 1] = 0
    y_line[y_line == 2] = 1

    fig = plt.figure(figsize = (5,5))
    ind = np.linspace(0, y_line.shape[-1] - 1, num = y_line.shape[-1])
    plt.text(0.05, 0.4, 'Test Accuracy: %.3f' % m.result().numpy())
    plt.plot(ind, l[:, -1], linewidth=1, color="red", label = "ground truth")
    plt.scatter(ind, y_line, alpha=0.1, label = "Prediction")
    plt.xlabel("SortedX")
    plt.ylabel("s")
    plt.legend()
    plt.title("Simple Neural Net")
    plt.show()

    fig = plt.figure(figsize = (5,5))
    sns.kdeplot(np.array(y_train[:, 0]), shade = True, color = 'red', legend = True)
    sns.kdeplot(np.array(y_test[:, 0]), shade = True, color = 'blue')
    sns.kdeplot(np.array(y_line), shade = True, color = 'turquoise')
    plt.ylabel("P(x)")
    plt.xlabel("x")
    plt.title(f"Neural Net Prediction Distribution")
    plt.show()
    return m.result().numpy()

if __name__ == "__main__":
    NeuralNet_Classifier()