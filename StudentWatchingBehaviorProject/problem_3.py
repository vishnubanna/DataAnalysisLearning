from problem_3_Polynomial_ridge import *
from problem_3_Simple_classifiers import *
from problem_3_Simple_NeuralNet import *

def main():
    """
    Apply all models required for problem 3

    Args: 
        None

    Returns: 
        None
    """
    Poly_ridge()
    Logistic_Classifier()
    GaussianBayes_Classifier()
    Knn_Classifier()
    RandomForrest_Classifier()
    NeuralNet_Classifier()

    # the models bellow will take a while to run 
    # so uncomment them and compile them if you absolutlye need too
    # SVM_Classifier()

    pass


if __name__ == "__main__":
    main()