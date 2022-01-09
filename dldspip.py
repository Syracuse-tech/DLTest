# Course test DSP-IP DL
# Iris dataset classification and data display
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets ,neighbors
import seaborn as sns
from matplotlib.colors import ListedColormap


# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Vars for plot - axes are Petal width vs Petal length and the respective classification
x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5

plt.figure(1, figsize=(8,6))
plt.clf()

# Plot the points - Classification markers as function of Petal width & length
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

####-------------------Part 2-------------------####
# 2.1
# I'll utilize K-Nearest Neighbors classifier

n_neighbors = 15
X = iris.data[:, :2]
y = iris.target
h = 0.02  # step size in the mesh

# Create color maps for plot
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]


# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=iris.target_names[y],
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",
)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(
    "Iris dataset classification with params: k = %i, weights = 'uniform'" % (n_neighbors)
)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.show()

#Part 2 - 2.2
# Using an svm estimator with SVC (rtb kernel) algorithm to train and predict classes

#Plot data aid function for grid
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C = 1.0
clf = svm.SVC(kernel="linear", C=C)
clf.fit(X,y)
plt.clf()
fig, sub = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plot_contours(sub, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
X0, X1 = X[:, 0], X[:, 1]
sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
sub.set_xlim(xx.min(), xx.max())
sub.set_ylim(yy.min(), yy.max())
sub.set_xlabel("Sepal length")
sub.set_ylabel("Sepal width")
sub.set_xticks(())
sub.set_yticks(())
sub.set_title("SVC with linear kernel")

plt.show()


print("D")