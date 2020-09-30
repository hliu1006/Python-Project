#!/usr/bin/env python
# coding: utf-8

# # Part 1: Using Data Science to Understand What Makes Wine Taste Good

# ## Section 1: Data Exploration
# 
# In this section, we'll do some exploratory analysis to understand the nature of our data and the underlying distribution.

# ### First, import some necessary libraries. 
# 
# ### Click the below cell block and run it.

# In[4]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for displaying DataFrames

import matplotlib.pyplot as plt
import seaborn as sns

# Import supplementary visualization code visuals.py from project root folder
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Usage of these libraries:
# * [numpy](http://www.numpy.org/) is a package in python that is used for scientific computing. It supports higher-order mathematical functions, higher dimensional arrays, matrices and other data structures.
# * [pandas](https://pandas.pydata.org/) is a very popular library that is used for a lot of data analysis and statistics related problems.
# * [time](https://docs.python.org/3.7/library/time.html) - standard module in python that allows for time related functions
# * [display](http://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html?highlight=display) is a module in the IPython toolkit that helps you display data structures in a nice, readable format.
# * [matplotlib](https://matplotlib.org/) is a very popular visualization library that lets you create a wide array of figures, charts and graphs in the IPython Notebook
# * [seaborn](https://seaborn.pydata.org/index.html) is another visualization tool that uses matplotlib underneath, and provides you with easy-to-use APIs for visualization. It also makes your graphs more prettier!

# ### Next, we'll load the dataset for red wines, and display the first 5 columns. Run the below cell block

# In[5]:


# TODO: Load the Red Wines dataset
data = pd.read_csv("data/winequality-red.csv", sep=';')
# TODO: Display the first five records
display(data.head(n=5))


# ## Now, let's do some basic preliminary analysis of our data:

# ### We'll begin by first seeing if our data has any missing information

# In[6]:


# TODO: Find if the data has any null information
data.isnull().any()


# ### Get additional information about the features in the data-set and their data types:

# In[7]:


#TODO: Get additional information about the data
data.info()


# #### The last column *quality* is a metric of how good a specific wine was rated to be. For our purposes, let's consider all wines with ratings 7 and above to be of very good quality, wines with 5 and 6 to be of average quality, and wines less than 5 to be of insipid quality.

# In[9]:


# Total number of wines
n_wines = data.shape[0]

# Number of wines with quality rating above 6
quality_above_6 = data.loc[(data['quality'] > 6)]
n_above_6 = quality_above_6.shape[0]

# TODO: Number of wines with quality rating below 5
quality_below_5 = data.loc[(data['quality'] < 5)]
n_below_5 = quality_below_5.shape[0]

# TODO: Number of wines with quality rating between 5 to 6
quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]

# Percentage of wines with quality rating above 6
greater_percent = n_above_6*100/n_wines

# Print the results

print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("Wines with rating less than 5: {}".format(n_below_5))
print("Wines with rating 5 and 6: {}".format(n_between_5))
print("Percentage of wines with quality 7 and above: {:.2f}%".format(greater_percent))


# ### Run the following cell block to see the distributions on a graph:

# In[11]:


# TODO: Visualize skewed continuous features of original data
vs.distribution(data, "quality")


# ### Get useful statistics, such as mean, median and standard deviation of the features:

# In[12]:


#TODO: Get some additional statistics, like mean, median and standard deviation
# Some more additional data analysis
display(np.round(data.describe()))


# As we can see, most fines fall under **average quality (between 5 and 6)**. Wines which were rated high are in the lower hundreds, whereas there are very few wines that aren't tasty enough (low ratings).
# 
# Next, since our aim is to predict the quality of wines, we’ll now extract the last column and store it separately.

# ## Section 2: Exploring Relationships between features

# In[13]:


#TODO: Draw a scatter-plot of features
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde');


# In[14]:


#TODO: Draw a heatmap between features
correlation = data.corr()
# display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[17]:


#Visualize the co-relation between pH and fixed Acidity

#Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, height=6)

#Draws a regression plot in the grid 
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.distplot)


# In[19]:


#TODO: Visualize a plot between Citric Acid levels and Fixed Acidity
#Create a new dataframe containing only Citric Acid and fixed acidity columns to visualize their co-relations
fixedAcidity_citricAcid = data[['citric acid', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridB = sns.JointGrid(x="fixed acidity", y="citric acid", data=fixedAcidity_citricAcid, height=6)

#Draws a regression plot in the grid 
gridB = gridB.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridB = gridB.plot_marginals(sns.distplot)


# In[21]:


# Visualize density vs fixed acidity
fixedAcidity_density = data[['density', 'fixed acidity']]
gridB = sns.JointGrid(x="fixed acidity", y="density", data=fixedAcidity_density, height=6)
gridB = gridB.plot_joint(sns.regplot, scatter_kws={"s": 10})
gridB = gridB.plot_marginals(sns.distplot)


# In[26]:


#Visualize quality vs volatile acidity
volatileAcidity_quality = data[['quality', 'volatile acidity']]
g = sns.JointGrid(x="volatile acidity", y="quality", data=volatileAcidity_quality, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)


# In[25]:


#We can visualize relationships of discreet values (quality vs volatile acidity) better with a bar plot

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='volatile acidity', data=volatileAcidity_quality, ax=axs)
plt.title('quality VS volatile acidity')

plt.tight_layout()
plt.show()
plt.gcf().clear()


# In[28]:


quality_alcohol = data[['alcohol', 'quality']]

g = sns.JointGrid(x="alcohol", y="quality", data=quality_alcohol, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)


# In[32]:


#TODO: Visualize quality vs alcohol with a bar plot
fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='alcohol', data=quality_alcohol, ax=axs)
plt.title('quality VS alcohol')

plt.tight_layout()
plt.show()
plt.gcf().clear()


# In[37]:


# TODO (OPTIONAL): Select any two features of your choice and view their relationship
quality_alcohol = data[['citric acid', 'residual sugar']]

g = sns.JointGrid(x="residual sugar", y="citric acid", data=quality_alcohol, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='residual sugar', y='citric acid', data=quality_alcohol, ax=axs)
plt.title('citric acid VS residual sugar')

plt.tight_layout()
plt.show()
plt.gcf().clear()


# ## Outlier Detection:
# 
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An **outlier step** is calculated as **1.5** times the **interquartile range (IQR)**. A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below:
# 
# * Assign the value of the 25th percentile for the given feature to Q1. Use np.percentile for this.
# * Assign the value of the 75th percentile for the given feature to Q3. Again, use np.percentile.
# * Assign the calculation of an outlier step for the given feature to step.
# * Optionally remove data points from the dataset by adding indices to the outliers list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!
# Once you have performed this implementation, the dataset will be stored in the variable good_data.

# In[38]:


#TODO: Find outliers for each feature

# For each feature find the data points with extreme high or low values
for feature in data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(data[feature], q=25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(data[feature], q=75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
    
    # OPTIONAL: Select the indices for data points you wish to remove
    outliers = []
    # Remove the outliers, if any were specified
    good_data = data.drop(data.index[outliers]).reset_index(drop = True)


# # Part 2: Using Machine Learning to Predict the Quality of Wines

# ### Data Preparation: 
# 
# ### First, we'll apply some transforms to convert our regression problem into a classification problem. Then, we'll use our data to create feature-set and target labels:

# In[46]:


#TODO: Convert the regression problem into a classification problem
"""
For our purposes, all wines with ratings less than 5 will fall under 0 (poor) category,
wines with ratings 5 and 6 will be classified with the value 1 (average), 
and wines with 7 and above will be of great quality (2).
"""
#Defining the splits for categories. 1–4 will be poor quality, 5–6 will be average, 7–10 will be great
bins = [1,4,6,10]
#0 for low quality, 1 for average, 2 for great quality
quality_labels=[0,1,2]
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)
display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])


# ### Next, shuffle and split our data-set into training and testing subsets:

# In[44]:


# Split the data into features and target label
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)
# Import train_test_split
from sklearn.model_selection import train_test_split

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
 quality_raw,
 test_size = 0.2,
 random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ## [scikit-learn](http://scikit-learn.org/) is a handy data science and machine learning library that lets you use ML algorithms in easy to use APIs.

# ###  Supervised Learning Models
# **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# ### Next, we will write a function that will accept a ML algorithm of our choice, and use our data to train it

# In[48]:


# Import two classification metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: quality training set
       - X_test: features testing set
       - y_test: quality testing set
    '''
    
    results = {}
    
    """
    Fit/train the learner to the training data using slicing with 'sample_size' 
    using .fit(training_features[:], training_labels[:])
    """
    start = time() # Get start time of training
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size]) #Train the model
    end = time() # Get end time of training
    
    # Calculate the training time
    results['train_time'] = end - start
    
    """
    Get the predictions on the first 300 training samples(X_train), 
    and also predictions on the test set(X_test) using .predict()
    """
    start = time() # Get start time
    predictions_train = learner.predict(X_train[:300])
    predictions_test = learner.predict(X_test)
    
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F1-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5, average='micro')
        
    # Compute F1-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='micro')
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!
# 
# Further reading: https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

# In[50]:


# TODO: Import any three supervised learning classification models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
clf_C = RandomForestClassifier(max_depth=None, random_state=None)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# TODO: Collect results on the learners
results = dict()
clfs = [clf_A, clf_B, clf_C]
samples = [samples_100, samples_10, samples_1]
for clf in clfs:
    clf_name = str(type(clf))
    results[clf_name] = dict()
    for i in range (len(samples)):
        results[clf_name][i] = train_predict_evaluate(clf, samples[i], X_train, y_train, X_test, y_test)
        
# TODO: Run metrics visualization for the three supervised learning models chosen using function in visuals.py
vs.visualize_classification_performance(results)


# ### Question: Why does Gaussian Naive Bayes perform poorly compared to the other methods?

# ### Answer:
# GNB assumes independence between features.
# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict the quality of wines.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the wines dataset.

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[51]:


# TODO: Import a supervised learning model that has 'feature_importances_'
model = RandomForestClassifier(max_depth=None, random_state=None)

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

print(X_train.columns)
print(importances)

# TODO: Plot importances
vs.feature_plot(importances, X_train, y_train)


# ## Hyperparameter tuning using GridSearchCV:

# In[52]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = RandomForestClassifier(max_depth=None, random_state=None)

# Create the parameters or base_estimators list you wish to tune, using a dictionary if needed.
# Example: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

"""
n_estimators: Number of trees in the forest
max_features: The number of features to consider when looking for the best split
max_depth: The maximum depth of the tree
"""
parameters = {'n_estimators': [10, 20, 30], 'max_features':[3,4,5, None], 'max_depth': [5,6,7, None]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5, average="micro")

# TODO: Perform grid search on the claszsifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
print("\nOptimized Model\n------")
print(best_clf)
print("\nFinal accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,  average="micro")))


# ## Finally, you can test out your model by giving it a bunch of inputs:

# In[53]:


"""Give inputs in this order: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol

"""
wine_data = [[8, 0.2, 0.16, 1.8, 0.065, 3, 16, 0.9962, 3.42, 0.92, 9.5],
            [8, 0, 0.16, 1.8, 0.065, 3, 16, 0.9962, 3.42, 0.92, 1 ],
            [7.4, 2, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 0.6]]
               
# Show predictions
for i, quality in enumerate(best_clf.predict(wine_data)):
    print("Predicted quality for Wine {} is: {}".format(i+1, quality))


# ## Question: What conclusions can you draw based on the above observations? Would you say that the model is more good at predicting average quality wines? Why?

# # TODOS:
# 1. Try solving this exercise again as a regression problem. Some of the common algorithms you can try from sklearn are *DecisionTreeRegressor*, *RandomForestRegressor*, and using *AdaBoostRegressor* with *DecisionTreeRegressor*. Some of the performance metrics that you might need to use in place of Accuracy and f1score are Mean Squared Error and R2Score
# 
# 2. Try using the White Wines data-set in place of the Red Wines
# 
