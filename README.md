
# Introduction to Pipelines

## Introduction

You've learned a substantial number of different supervised learning algorithms. Now, it's time to learn about a handy tool used to integrate these algorithms into a single manageable pipeline.

## Objectives

You will be able to:

- Explain how pipelines can be used to test many different parameters 

## Why Use Pipelines?

Pipelines are extremely useful tools to write clean and manageable code for machine learning. Recall how we start preparing our dataset: we want to clean our data, transform it, potentially use feature selection, and then run a machine learning algorithm. Using pipelines, you can do all these steps in one go!

Pipeline functionality can be found in the scikit-learn's `Pipeline` module. Pipelines can be coded in a very simple way:

```python
from sklearn.pipeline import Pipeline
   
pipe = Pipeline([('mms', MinMaxScaler()),
                 ('tree', DecisionTreeClassifier(random_state=123))
                 ('RF', RandomForestClassifier(random_state=123))])
```

This pipeline will ensure that when running the model on our data, first we'll apply a MinMax scaler on our features. Next, a decision tree is applied to the data. Finally, we also fit a random forest to the data. 

The model(s) can be fit using: 

```python
pipe.fit(X_train, y_train)

```

A really good blogpost on the basic ideas of pipelines can be found [here](https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html).


## Integrating Grid Search in Pipelines

Note that the above pipeline simply creates one pipeline for a training set, and evaluates on a test set. Is it possible to create a pipeline that performs grid search? And Cross-Validation? Yes, it is!

First, you define the pipeline same way as above. Next, you create a parameter grid. When this is all done, you use the function `GridSearchCV()`, which you've seen before, and specify the pipeline as the estimator and the parameter grid. You also have to define how many folds you'll use in your cross-validation. 

```python
# Create the pipeline
pipe = Pipeline([('mms', MinMaxScaler()),
                 ('tree', DecisionTreeClassifier(random_state=123))
                 ('RF', RandomForestClassifier(random_state=123))])

# Create the grid parameter
grid = [{'tree__max_depth': [None, 2, 6, 10], 
         'tree__min_samples_split': [5, 10], 
         'RF__max_depth': [None, 2, 3, 4, 5, 6], 
         'RF__min_samples_split': [2, 5, 10]}]


# Create the grid, with "pipe" as the estimator
gridsearch = GridSearchCV(estimator=pipe, 
                          param_grid=grid, 
                          scoring='accuracy', 
                          cv=3)

# Fit using grid search
gridsearch.fit(X_train, y_train)
```

An article with a detailed workflow can be found [here](https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-2.html).

## Summary

Great, this wasn't too difficult! The proof of all this is in the pudding. In the next lab, you'll extensively use this workflow to build several pipelines applying several classification algorithms. 
