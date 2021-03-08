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

