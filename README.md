# MachineLearningMastery

## Day 1 - Machine Learning Mastery:

Installation of Anaconda, ( which includes python and required default packages installed in Anaconda IDE).

Checking the installed versions in jupyter notebook.

## Day 2 - Machine Learning Mastery:

Creating a Dataframe

## Day 3 - Machine Learning Mastery:

Load data from CSV files ( use standard ML datasets)

UCI machine learning repository: 
http://archive.ics.uci.edu/ml/index.php

## Day 4 - Machine Learning Mastery:

First step to **understanding your data** is to use **descriptive statistics**.

Learn how to use descriptive statistics to understand your data. I recommend using the **helper functions** provided on the **Pandas** DataFrame.

--> Understand your data using the **head()** function to look at the first few rows.

--> Review the dimensions of your data with the **shape** property.

--> Look at the data types for each attribute with the **dtypes** property.

--> Review the distribution of your data with the **describe()** function.

--> Calculate pair-wise correlation between your variables using the **corr()** function.

## Day 5 - Machine Learning Mastery:

Second way to improve your **understanding of your data** is by using **data visualization** techniques (e.g. plotting).

Learn how to use plotting in Python to understand **attributes** alone and their interactions. Again, I recommend using the **helper functions** provided on the Pandas DataFrame.

--> Use the **hist()** function to create a histogram of each attribute.

--> Use the **plot(kind=’box’)** function to create box-and-whisker plots of each attribute.

--> Use the **pandas.scatter_matrix()** function to create pairwise scatterplots of all attributes.

## Day 6 - Machine Learning Mastery:

Your raw data may not be setup to be in the best shape for modeling.

Sometimes you need to preprocess your data in order to best present the inherent structure of the problem in your data to the modeling algorithms. 
In today’s lesson, you will use the pre-processing capabilities provided by the scikit-learn.

The scikit-learn library provides two standard idioms for transforming data. Each transform is useful in different circumstances: 
**Fit and Multiple Transform** and **Combined Fit-And-Transform**.

There are many techniques that you can use to prepare your data for modeling. For example, try out some of the following

* Standardize numerical data (e.g. mean of 0 and standard deviation of 1) using the scale and center options.
* Normalize numerical data (e.g. to a range of 0-1) using the range option.
* Explore more advanced feature engineering such as Binarizing.

For example, the snippet below loads the Pima Indians onset of diabetes dataset, calculates the parameters needed to standardize the data, then creates a standardized copy of the input data.

## Day 7 - Machine Learning Mastery:

**Algorithm Evaluation With Resampling Methods**

The dataset used to train a machine learning algorithm is called a training dataset. The dataset used to train an algorithm cannot be used to give you reliable estimates of the accuracy of the model on new data. This is a big problem because the whole idea of creating the model is to make predictions on new data.

You can use statistical methods called resampling methods to split your training dataset up into subsets, some are used to train the model and others are held back and used to estimate the accuracy of the model on unseen data.

Your goal with today’s lesson is to practice using the different resampling methods available in scikit-learn, for example:

* Split a dataset into training and test sets.
* Estimate the accuracy of an algorithm using k-fold cross validation.
* Estimate the accuracy of an algorithm using leave one out cross validation.

The snippet below uses scikit-learn to estimate the accuracy of the Logistic Regression algorithm on the Pima Indians onset of diabetes dataset using 10-fold cross validation.

## Day 8 - Machine Learning Mastery:

**Algorithm Evaluation Metrics**

There are many different metrics that you can use to evaluate the skill of a machine learning algorithm on a dataset.

You can specify the metric used for your test harness in scikit-learn via the **cross_validation.cross_val_score()** function and defaults can be used for regression and classification problems. Your goal with today’s lesson is to practice using the different algorithm performance metrics available in the scikit-learn package.

* Practice using the Accuracy and LogLoss metrics on a classification problem.
* Practice generating a confusion matrix and a classification report.
* Practice using RMSE and RSquared metrics on a regression problem.

The snippet below demonstrates calculating the LogLoss metric on the Pima Indians onset of diabetes dataset.

## Day 9 - Machine Learning Mastery:

**Spot-Check Algorithms**

You cannot possibly know which algorithm will perform best on your data beforehand.

You have to discover it using a process of trial and error. I call this spot-checking algorithms. 
The scikit-learn library provides an interface to many machine learning algorithms and tools to compare the estimated 
accuracy of those algorithms.

In this lesson, you must practice spot checking different machine learning algorithms.

* Spot check linear algorithms on a dataset (e.g. linear regression, logistic regression and linear discriminate analysis).
* Spot check some non-linear algorithms on a dataset (e.g. KNN, SVM and CART).
* Spot-check some sophisticated ensemble algorithms on a dataset (e.g. random forest and stochastic gradient boosting).

For example, the snippet below spot-checks the K-Nearest Neighbors algorithm on the Boston House Price dataset.

## Day 10 - Machine Learning Mastery:

**Model Comparison and Selection**

Now that you know how to spot check machine learning algorithms on your dataset, you need to know how to compare the estimated performance of different algorithms and select the best model.

You will practice comparing the accuracy of machine learning algorithms in Python with scikit-learn.

* Compare linear algorithms to each other on a dataset.
* Compare nonlinear algorithms to each other on a dataset.
* Compare different configurations of the same algorithm to each other.
* Create plots of the results comparing algorithms.
* 
The example below compares Logistic Regression and Linear Discriminant Analysis to each other on the Pima Indians onset of diabetes dataset.

## Day 11 - Machine Learning Mastery:

**Improve Accuracy with Algorithm Tuning**

Once you have found one or two algorithms that perform well on your dataset, you may want to improve the performance of those models.

One way to increase the performance of an algorithm is to tune its parameters to your specific dataset.

The scikit-learn library provides two ways to search for combinations of parameters for a machine learning algorithm. Your goal in today’s lesson is to practice each.

* Tune the parameters of an algorithm using a grid search that you specify.
* Tune the parameters of an algorithm using a random search.

The snippet below uses is an example of using a grid search for the Ridge Regression algorithm on the Pima Indians onset of diabetes dataset.

## Day 12 - Machine Learning Mastery:

**Improve Accuracy with Ensemble Predictions**

Another way that you can improve the performance of your models is to combine the predictions from multiple models.

Some models provide this capability built-in such as random forest for bagging and stochastic gradient boosting for boosting. Another type of ensembling called voting can be used to combine the predictions from multiple different models together.

In this lesson, you will practice using ensemble methods.

* Practice bagging ensembles with the random forest and extra trees algorithms.
* Practice boosting ensembles with the gradient boosting machine and AdaBoost algorithms.
* Practice voting ensembles using by combining the predictions from multiple models together.

The snippet below demonstrates how you can use the Random Forest algorithm (a bagged ensemble of decision trees) on the Pima Indians onset of diabetes dataset.

## Day 13 - Machine Learning Mastery:

**Finalize And Save Your Model**

Once you have found a well-performing model on your machine learning problem, you need to finalize it.

In this lesson, you will practice the tasks related to finalizing your model.

* Practice making predictions with your model on new data (data unseen during training and testing).
* Practice saving trained models to file and loading them up again.

For example, the snippet below shows how you can create a Logistic Regression model, save it to file, then load it later and make predictions on unseen data.

## Day 14 - Machine Learning Mastery:

HelloWorld Project


**Source Link:**

**https://machinelearningmastery.com/python-machine-learning-mini-course/**
