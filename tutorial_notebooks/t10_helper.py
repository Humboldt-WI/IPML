''' Helper function for the tenth tutorial notebook on decision trees. '''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def generate1D_data(n=5_000, b=[-6, 3], seed=501, plot=True):
	'''
	Generates 1D synthetic data for binary classification with an option to
 	plot the dataset.

	This function creates a dataset consisting of a single feature and a
	binary target variable, simulating a logistic regression scenario. The
	probability of each sample belonging to class 1 is determined by applying
	the logistic function to a linear combination of the input feature.

	Parameters:
	- n (int, optional): Number of samples to generate. Defaults to 5000.
	- b (list of floats, optional): Coefficients for the linear combination,
 		where `b[0]` is the intercept and `b[1]` the slope. Defaults to [-6, 3].
	- seed (int, optional): Random seed for reproducibility. Defaults to 501.
	- plot (bool, optional): If True, plots the generated data. Defaults to True.

	Returns:
	- x (numpy.ndarray): The generated feature, shaped as (n_samples, 1).
	- y (numpy.ndarray): The binary target variable, shaped as (n_samples,).
	- p (numpy.ndarray): The probability of each sample belonging to class 1,
 		shaped as (n_samples,).

	Notes:
	- The feature values are drawn from a normal distribution centered at 2
 		with a standard deviation of 1.
	- The target variable is sampled from a binomial distribution, where the
 		probability of success for each trial is given by applying the logistic
   		function to the linear combination of the feature.

	Example:
	>>> x, y, p = generate1D_data(n=1000, b=[-2, 1], seed=42)
	Generates 1000 samples with a linear combination defined by b=[-2, 1],
 	using a seed of 42. If `plot` is True, the function also displays a
  	scatter plot of the generated data.
	'''

	np.random.seed(seed)
	x = np.random.normal(2, 1, n).reshape((-1, 1))  # feature
	z = b[0] + b[1] * x  # latent variable
	p = 1 / (1 + np.exp(-z))  # probability (of sample being in class 1)
	y = np.random.binomial(1, p=p).flatten()  # sampled class assignment

	if plot:
		plt.scatter(x, y, alpha=0.05)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

	return x, y, p


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


def predict_plot(model, x_true, y_true, lims=None, proba=False, axis=plt):
    '''
    Plots the true and predicted values from a model, optionally including predicted
    probabilities.

    This function visualizes the performance of a fitted model by comparing its
    predictions to the true values. It supports plotting for both regression and
    classification tasks, with an option to display predicted probabilities for
    classification models.

    Parameters:
    - model (object): The machine learning model with a `predict` method and,
    	if `proba` is True, a `predict_proba` method.
    - x_true (array-like): Actual input values, expected to be in the shape (n_samples, 1).
    - y_true (array-like): Actual target values, with shape (n_samples,) or (n_samples, 1).
    - lims (tuple of tuple, optional): Axis limits as ((x_min, x_max), (y_min, y_max)).
    	Defaults to None, auto-setting the limits.
    - proba (bool, optional): If True, plots predicted probabilities instead of values.
    	Defaults to False.
    - axis (matplotlib axis, optional): The plotting axis to use. Defaults to
    	matplotlib.pyplot.

    Returns:
    None. Displays a plot of the true versus predicted values or probabilities.

    Notes:
    - The function automatically rounds the min and max of `x_true` to the nearest
    	hundredth for plotting.
    - It uses alpha blending to differentiate the true points from the predicted
    	line or curve.
    - When `proba` is True, the model must implement `predict_proba`, or an error
    	will occur.

    Example:
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([2, 4, 6])
    >>> model = LinearRegression().fit(X, y)
    >>> predict_plot(model, X, y)
    Displays a plot with true values and predicted values.
    '''

    grid = np.arange(x_true.min().round(
    	2), x_true.max().round(2), 0.005).reshape(-1, 1)

    axis.scatter(x_true, y_true, alpha=0.2, label="True")
    axis.plot(grid, model.predict(grid), label="Predicted",
              c="darkorange", linewidth=3)

    if proba:
        axis.plot(
            grid,
            model.predict_proba(grid)[:, 1],
            label="Predicted Probability",
            c="darkgreen",
            linewidth=3,
        )

    title = type(model).__name__

    if axis == plt:
        axis.xlabel("x")
        axis.ylabel("y")
        axis.legend()
        axis.title(title)
        if lims:
            axis.xlim(lims[0])
            axis.ylim(lims[1])
        axis.show()
    else:
        axis.set(xlabel='x', ylabel='y')
        axis.legend()
        axis.set_title(title)
        if lims:
            axis.set_xlim(lims[0])
            axis.set_ylim(lims[1])


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
    ix = y == 1
    ax1.scatter(X[ix, 0], X[ix, 1], c='blue',  label="Positives")
    ax1.scatter(X[~ix, 0], X[~ix, 1], c='red',  label="Negatives")
    ax1.set(xlabel="$X_1$", ylabel="$X_2$")
    ax1.legend(loc='best')

    learner.fit(X, y)
    x_1, x_2 = np.mgrid[-2:8:resolution, -2:8:resolution]
    grid = np.column_stack((x_1.ravel(), x_2.ravel()))
    model_probs = learner.predict(grid).reshape(x_1.shape)

    model_name = type(learner).__name__
    
    ax2.set_title(model_name)
    ax2.contourf(x_1, x_2, model_probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax2.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu",
                vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
    ax2.set(xlabel="$X_1$", ylabel="$X_2$")

    return fig, ax1, ax2
