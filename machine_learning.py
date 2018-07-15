# Blog link: http://blog.kaggle.com/author/kevin-markham/
import csv
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, \
                                    KFold, cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt


def load_dataset_manually():
    with open('iris.data', 'r') as inputfile:
        reader = csv.reader(inputfile)
        for index, row in enumerate(reader):
            print (row)


def load_dataset_and_analyse():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn1 = k_nearest(X, y, 1)
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    knn1.predict(X_new)  # Returns 2,1

    knn5 = k_nearest(X, y, 5)
    knn5.predict(X_new)  # Returns 1, 1

    # logreg = logreg_prediciting(X,y)
    # logreg.predict(X_new)  # Returns 2,0

    # print (metrics.accuracy_score(y, knn5.predict(X)))  # Training accuracy
    # print (metrics.accuracy_score(y, knn1.predict(X)))  # Training accuracy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
    # 0.4 means test size is 40% of the original data. Standard for it is around 20-40%
    # Random_state helps split the data same way every time. Without it, it will split it differently everytime
    knn5 = k_nearest(X_train, y_train, 5)

    print (metrics.accuracy_score(y_test, knn5.predict(X_test)))  # Testing accuracy

    # Can we locate an even better value for K?
    scores = []
    for k in range(1, 26): # Testing K = 1 to 25
        knn = k_nearest(X_train, y_train, k)
        scores.append(metrics.accuracy_score(y_test, knn.predict(X_test)))

    # Cross validation example
    # Simulate splitting a dataset of 25 observations into 5 folds
    kf = KFold(25, n_folds=5, shuffle=False)

    # 1. Dataset contains 25 observations (numbered 0 through 24)
    # 2. 5 fold cross validation, thus it runs for 5 iterations
    # 3. For each iteration, every observation is either in the
    # training set or testing set but not both
    # 4. Every observation is in the testing set exactly once

    # Print the contents of each training and testing set
    print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
    for iteration, data in enumerate(kf, start=1):
        print('{} {} {}'.format(iteration, data[0], data[1]))


    # 10 fold cross validation with k=5 for knn
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # cv=10 means 10 fold cross validation
    # scoring='accuracy' classification accuracy as the evaluation metrics

    print(scores)

    # use average accuracy as an estimate of out of sample accuracy
    print(scores.mean())

    # Search for an optimal value of k for knn
    k_scores = []
    for k in range(1, 31):  # Testing K = 1 to 30
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print(k_scores)

    plt.plot(range(1,31), k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    # K = 20 should be picked from this graph even though
    # K =13, 18 and 20 have the same highest accuracy of 0.98.
    # This is because we want our models to be simples
    # and higher k values means less complexity

    # 10 fold cross validation with logistics regression
    logreg = LogisticRegression()
    print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
    # 0.95333
    # It means knn = 20 is a better fit than logreg

    # The above strategy of using a for loop to find the optimal value of K
    # can be done through GridSearchCV. It replaces the for loop and provides
    # addtional functionality

    # Define the values that should be searched
    k_range = range(1, 31)

    # Create a param grid: map the paramter names to the values that should
    # be searched
    param_grid = dict(n_neighbors=k_range)
    knn = KNeighborsClassifier()
    # instantiate the grid
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    # Set n_jobs = -1 to run computations in parallel
    # (if your computer and OS allows it)

    grid.fit(X, y)  # This step can take a while depending on the model and data

    # view the complete results (list of named tuples)
    grid.grid_scores_
    # [mean: 0.96, std: 0.0533, params: {'n_neighbors': 1},
    #  mean: 0.9533, std: 0.05207, params: {'n_neighbors': 2}, ..]

    grid.grid_scores_[0].parameters
    grid.grid_scores_[0].cv_validation_scores
    grid.grid_scores_[0].mean_validation_score

    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    plt.plot(k_range, grid_mean_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    # plotting a graph isnt the most efficient way of finding the optimal k value

    # examine the best model
    print(grid.best_score_)  # best accuracy
    print(grid.best_params_)  # best param used for that accuracy
    print(grid.best_estimator_)  # best model used for the param

    weight_options = ['uniform', 'distance']
    # Another param of knn that can be tuned is the weights
    # Default value is uniform which means it puts uniform weight into all the
    # k neighbour. Distance is another option where the closer neighbours are
    # weighted more than further neighbours

    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    # examine the best model
    print(grid.best_score_)  # 0.98
    print(grid.best_params_)  # {'n_neighbors': 13, 'weights' : 'uniform'}
    # Distance on knn didnt improve over uniform

    # train your model using all the data and best known parameters
    knn = KNeighborsClassifier(n_neighbour=13, weights='uniform')
    knn.fit(X, y)
    knn.predict([3, 5, 4, 2])  # predict out of sample data

    # Shortcut: grid can do the prediction
    grid.predict([3, 5, 4, 2])

    # Reducing computational expense using RandomizedSearchCV
    # RandomizedSearchCV is close cousin of GridSearchCV
    # RandomizedSearchCV seaches a subset of the parameters
    # and you control the computational "budget"

    # Specify "parameter distn" rather than "parameter grid"
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    # Important: If one of your tuning parameters is continous, Specify
    # a continous distn rather than a list of values

    # n_iter controls the number of searches
    # random_state is there for the purpose of reproducability
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy',
                              n_iter=10, random_state=5)
    rand.fit(X, y)
    rand.grid_scores_
    print(rand.best_score_)
    print(rand.best_params_)


def k_nearest(X, y, k):
    # Four conditions for sklearn
    # 1. Features and  response should be separate objects
    # 2. X and y should have numeric data
    assert type(X[0, 0]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(X[-1, -1]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(y[0]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(y[-1]) in [np.float64, np.float32, np.int64, np.int32]

    # 3. Features and response should be numpy arrays
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray

    # 4. Features and response should have specific shapes
    assert len(y.shape) == 1  # Only 1 column in the target
    assert y.shape[0] == X.shape[0]
    # Number of rows match between data and target

    knn = KNeighborsClassifier(n_neighbors=k)  # Using k neighbour(s)
    knn.fit(X, y)  # In place

    return knn

def logreg_prediciting(X, y):
    logreg = LogisticRegression()
    logreg.fit(X,y)
    return logreg

def linreg_predicting(X, y):
    linreg = LinearRegression()
    linreg.fit(X, y)
    return linreg

def working_with_pandas():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
    print(data.shape)  # (200, 4)

    '''
    What are the features?
    TV: advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
    Radio: advertising dollars spent on Radio
    Newspaper: advertising dollars spent on Newspaper
    What is the response?
    Sales: sales of a single product in a given market (in thousands of widgets)

    Because response variable is continous, this is a regression problem

    '''
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # Default split is 75% for training and 25% for testing

    linreg = linreg_predicting(X_train, y_train)
    print(linreg.intercept_)
    print(linreg.coef_)

    y_pred = linreg.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))  # MAE
    print(metrics.mean_squared_error(y_test, y_pred))  # MSE
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  # RMSE
    #  1.40465

    # If you plot the data, you will see that newspaper has a very weak linear correlation to sales
    # So, lets try removing it and re running our predictions

    X = data[['TV', 'Radio']]
    y = data['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # Default split is 75% for training and 25% for testing

    linreg = linreg_predicting(X_train, y_train)
    print(linreg.intercept_)
    print(linreg.coef_)

    y_pred = linreg.predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_pred))  # MAE
    print(metrics.mean_squared_error(y_test, y_pred))  # MSE
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  # RMSE
    #  1.388 <= Slight decrease from above
    # Which means newspaper feature in unlikely to be useful for predictiing sales and should be removed

    # Using cross validation to do feature selection i.e whether newspaper should be included or not
    lm = LinearRegression()

    # 10 fold cross-validation with all three features
    scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
    # scoring = 'accuracy' is only relevant for classification problems
    # scoring = 'root_mean_squared' is best but it is not avaliable in cross_val_score
    # So we will use mean_squared_error and then take the sqrt later
    print(scores)

    # fix the sign of MSE scores
    mse_scores = -scores
    print(mse_scores)

    # convert from MSE to RMSE
    rmse_scores = np.sqrt(mse_scores)
    print(rmse_scores)

    print(rmse_scores.mean())
    # 1.69

    # 10 fold cross validation with 2 features (excluding newspaper)
    X = data[['TV', 'Radio']]
    print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean())
    # 1.68
    # Since 1.68 < 1.69 and we are trying to minimize RMSE, it means
    # that model excluding newspaper is the better model

def video9lesson():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv(url, header=None, names=col_names)

    # define X and y
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
    X = pima[feature_cols]
    y = pima.label

    # split X and y into training and testing sets
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # train a logistic regression model on the training set
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    # calculate accuracy
    print(metrics.accuracy_score(y_test, y_pred_class)) # 0.692708333333

    # Null accuracy: accuracy that could be achieved by always
    # predicting the most frequent class i.e a "dumb" model
    # calculate null accuracy (for binary classification problems coded as 0/1)
    max(y_test.mean(), 1 - y_test.mean()) # 0.677
    # Our model accuracy of 0.69 is terrible compared to the null accuracy of 0.677

    # calculate null accuracy (for multi-class classification problems)
    y_test.value_counts().head(1) / len(y_test)

    # print the first 25 true and predicted responses
    print('True:', y_test.values[0:25]) # [1 0 0 1 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 0 0]
    print('Pred:', y_pred_class[0:25])  # [0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # This tell us that we often correctly predict the 0 case but we rarely accurately predict the 1 case

    # This is a limitation of classification accuracy
    # Classification accuracy is the easiest classification metric to understand but
    # it does not tell you the underlying distribution of response values and
    #  it does not tell you what "types" of errors your classifier is making

    # Enter confusion matrix

    # IMPORTANT: first argument is true values, second argument is predicted values
    print(metrics.confusion_matrix(y_test, y_pred_class))
    # [[118  12]
    # [ 47  15]]
    '''
    Basic terminology
    True Positives (TP): we correctly predicted that they do have diabetes
    True Negatives (TN): we correctly predicted that they don't have diabetes
    False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
    False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")

    N = 192
    TN = 118/192, TP = 15/192, FP = 12/192, FN = 47/192
    '''

    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # Classification Accuracy: Overall, how often is the classifier correct?
    print((TP + TN) / float(TP + TN + FP + FN)) # 0.6927
    print(metrics.accuracy_score(y_test, y_pred_class)) # 0.6927 <= same number as above

    # Classification Error: Overall, how often is the classifier incorrect?
    # Also known as "Misclassification Rate"
    print((FP + FN) / float(TP + TN + FP + FN))  # 0.30729
    print(1 - metrics.accuracy_score(y_test, y_pred_class)) # 0.30729

    # Sensitivity: When the actual value is positive, how often is the prediction correct?
    # Also known as "True Positive Rate" or "Recall"
    print(TP / float(TP + FN))  # 0.24194
    print(metrics.recall_score(y_test, y_pred_class))  # 0.24194

    # Specificity: When the actual value is negative, how often is the prediction correct?
    print(TN / float(TN + FP))  # 0.90769
    # No metrics function for this

    # False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    # Also 1 - Specificity
    print(FP / float(TN + FP))  # 0.0923

    # Precision: When a positive value is predicted, how often is the prediction correct?
    print(TP / float(TP + FP))  # 0.55555
    print(metrics.precision_score(y_test, y_pred_class))  # 0.55555

    # Many other metrics can be computed: F1 score, Matthews correlation coefficient, etc.

    # print the first 10 predicted responses
    logreg.predict(X_test)[0:10]  #  [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

    # print the first 10 predicted probabilities of class membership
    logreg.predict_proba(X_test)[0:10, :]
    '''
    [[ 0.63247571,  0.36752429],
       [ 0.71643656,  0.28356344],
       [ 0.71104114,  0.28895886],
       [ 0.5858938 ,  0.4141062 ],
       [ 0.84103973,  0.15896027],
       [ 0.82934844,  0.17065156],
       [ 0.50110974,  0.49889026],
       [ 0.48658459,  0.51341541],
       [ 0.72321388,  0.27678612],
       [ 0.32810562,  0.67189438]]
    '''

    # print the first 10 predicted probabilities for class 1 only
    logreg.predict_proba(X_test)[0:10, 1]
    # [ 0.36752429,  0.28356344,  0.28895886,  0.4141062 ,  0.15896027,
    # #0.17065156,  0.49889026,  0.51341541,  0.27678612,  0.67189438]

    # store the predicted probabilities for class 1
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]

    # histogram of predicted probabilities
    plt.hist(y_pred_prob, bins=8)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of diabetes')
    plt.ylabel('Frequency')
    plt.show()

    # Decrease the threshold for predicting diabetes in order
    # to increase the sensitivity of the classifier

    # predict diabetes if the predicted probability is greater than 0.3
    from sklearn.preprocessing import binarize
    y_pred_class = binarize([y_pred_prob], 0.3)[0]

    # print the first 10 predicted probabilities
    y_pred_prob[0:10]
    # [ 0.36752429,  0.28356344,  0.28895886,  0.4141062 ,  0.15896027,
    #   0.17065156,  0.49889026,  0.51341541,  0.27678612,  0.67189438]

    # print the first 10 predicted classes with the lower threshold
    y_pred_class[0:10]
    # [ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.]

    # previous confusion matrix (default threshold of 0.5)
    print(confusion)
    # [[118  12]
    # [ 47  15]]

    # new confusion matrix (threshold of 0.3)
    print(metrics.confusion_matrix(y_test, y_pred_class))
    # [[80 50]
    # [16 46]]

    # sensitivity has increased (used to be 0.24)
    print(46 / float(46 + 16))  # 0.74193

    # specificity has decreased (used to be 0.91)
    print(80 / float(80 + 50))  # 0.61538

    '''
    Threshold of 0.5 is used by default (for binary problems) to convert predicted probabilities into class predictions
    Threshold can be adjusted to increase sensitivity or specificity
    Sensitivity and specificity have an inverse relationship
    '''

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(thresholds, threshold):
        print('Sensitivity:', tpr[thresholds > threshold][-1])
        print('Specificity:', 1 - fpr[thresholds > threshold][-1])

    evaluate_threshold(thresholds, 0.5)
    # Sensitivity: 0.241935483871
    # Specificity: 0.907692307692

    evaluate_threshold(thresholds, 0.3)
    # Sensitivity: 0.725806451613
    # Specificity: 0.615384615385

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print(metrics.roc_auc_score(y_test, y_pred_prob))  # 0.72457

    # calculate cross-validated AUC
    from sklearn.cross_validation import cross_val_score
    cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()  # 0.73782


if __name__ == "__main__":
    # load_dataset_and_analyse()
    # working_with_pandas()
    video9lesson()