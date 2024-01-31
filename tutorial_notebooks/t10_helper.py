import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def plot_class_boundary(X, y, learner, resolution=0.1):
    '''
    Plot the given dataset and the decision boundaries of a classifier.

    This function creates a two-panel plot. The first panel shows the given dataset with two classes, 
    and the second panel shows the decision boundaries or predictions of the specified classifier. 
    The classifier is trained on the dataset within this function.

    Parameters:
    - X (array-like): Feature dataset. It should be a 2D array with shape [n_samples, n_features].
    - y (array-like): Labels for the dataset. It should be a 1D array with shape [n_samples].
    - learner (classifier object): A classifier object that implements `fit` and `predict` methods.
    - resolution (float, optional): The resolution of the grid for plotting the decision boundaries. 
      Defaults to 0.1.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    - ax1 (matplotlib.axes._subplots.AxesSubplot): The subplot showing the dataset.
    - ax2 (matplotlib.axes._subplots.AxesSubplot): The subplot showing the classifier's decision boundaries.

    Notes:
    - The function assumes binary classification and uses blue and red colors to represent the two classes.
    - The classifier's decision boundaries are plotted using a contour plot.

    Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = generate2D_data(300)  # assuming generate2D_data is defined
    >>> fig, ax1, ax2 = plot_classifier_solution(X, y, LogisticRegression())
    >>> plt.show()
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    ax1.set_title('Data set')
    ix = y==1
    ax1.scatter(X[ix,0], X[ix,1], c='blue',  label="Positives")
    ax1.scatter(X[~ix,0], X[~ix,1], c='red',  label="Negatives")
    ax1.set(xlabel="$X_1$", ylabel="$X_2$")
    ax1.legend(loc='best')

    learner.fit(X, y)
    x_1, x_2 = np.mgrid[-2:8:resolution, -2:8:resolution]
    grid = np.column_stack((x_1.ravel(), x_2.ravel()))
    model_probs = learner.predict(grid).reshape(x_1.shape)
    
    ax2.set_title('Classifier')
    ax2.contourf(x_1, x_2, model_probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax2.scatter(X[:,0], X[:,1], c=y, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
    ax2.set(xlabel="$X_1$", ylabel="$X_2$")

    return fig, ax1, ax2


def generate2D_data(size=500):
    '''
    Generate a 2D dataset with two clusters using the `make_blobs` function.

    The function creates a 2D dataset with two predefined mean points, each forming a cluster. 
    This is useful for visualizing the decision boundary of a classification model or other machine
    learning tasks such as clustering. 

    Parameters:
    - size (int, optional): The total number of samples to generate. 
      Defaults to 500 if not specified.

    Returns:
    - X (array of shape [n_samples, 2]): The generated samples.
    - y (array of shape [n_samples]): The integer labels for cluster membership of each sample.

    Notes:
    - The clusters are centered around the points (1, 1) and (4, 4).
    - The function sets a random state for reproducibility of the data generation.

    Example:
    >>> X, y = generate2D_data(300)
    >>> print(X.shape)
    (300, 2)
    >>> print(y.shape)
    (300,)
    '''
    means = [[1, 1], [4, 4]]
    X, y = make_blobs(n_samples=size, centers=means, random_state=888)
    return X, y
