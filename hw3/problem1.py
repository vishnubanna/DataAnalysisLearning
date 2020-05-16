import numpy as np
import matplotlib.pyplot as plt
from helper import *


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities

    :param hist: list
    :return: list
    """

    items = np.sum(np.asarray(hist))
    return hist/items


def computeJ(histo, width):
    """
    calculate computeJ for one bin width

    :param histo: list
    :param width: int
    :return: float
    """
    #if (np.size(histo) - 1 != 0 ):
    items = np.sum(histo)
    j = None
    j_1 = (2/((items - 1) * width))
    j = ((items + 1)/((items - 1) * width)) * np.sum(np.square(histo/items))
    j = j_1 - j
    #print(j, np.sum(np.square(histo/items)))
        
    #else:
        #j = float('inf')

    # if(j == np.nan){
    #     j = float('inf')
    # }
    return j


def sweepN (data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate computeJ for a full sweep [min_bins to max_bins]

    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """

    print(data)
    print(minimum, maximum)
    print(min_bins, max_bins)
    width = None
    histo = None
    j = []

    for i in range(min_bins, max_bins + 1):
        histo = plt.hist(data, bins = i)
        width = (maximum - minimum)/i
        j.append(computeJ(histo[0], width))
        #print(i, j[i-1])
    
    #minim = findMin(j)
    return j


def findMin (l):
    """
    generic function that takes a list of numbers and returns smallest number in that list its index.
    return optimal value and the index of the optimal value as a tuple.

    :param l: list
    :return: tuple
    """

    minim = min(l)
    for i, number in enumerate(l):
        if number == minim:
            return (minim, i)
    return (minim, None)
    


if __name__ == '__main__':
    data = getData()  # reads data from inp.txt. This is defined in helper.py
    lo = min(data)
    hi = max(data)

    
    js = sweepN(data, lo, hi, 1, 100)

    # the values 1 and 100 represent the lower and higher bound of the range of bins.
    # They will change when we test your code and you should be mindful of that.
    
    print(findMin(js))
    j = findMin(js)

    fig = plt.figure()
    histofull = plt.hist(data, bins = j[1] + 1, density = True)
    histdata = norm_histogram(histofull[0])
    print(histofull, histdata)
    plt.title("Histogram bins = 15")
    plt.ylabel("count")
    plt.xlabel("bins")
    #plt.show()
    #plotHisto(histo, "prog.png")




    # Include code here to plot js vs. the bin range

