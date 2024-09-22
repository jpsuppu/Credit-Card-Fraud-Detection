#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing and importing the necessary packages

get_ipython().system('pip install --user imbalanced-learn scikit-learn')
# pip install --upgrade --user imbalanced-learn scikit-learn
get_ipython().system('pip install --user imbalanced-learn')
get_ipython().system('pip install xgboost')
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')
import sklearn 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, log_loss,roc_curve, auc,RocCurveDisplay, PrecisionRecallDisplay,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[2]:


# Import the dataset from a CSV file
# Ensure the file path is correctly specified to your local system
# In this case, the dataset is stored in a directory under OneDrive

dataset=pd.read_csv(r'C:\Users\znpr9756\OneDrive - orange.com\Documents\Data science\creditcard.csv')


# Display the first few rows of the dataset to verify it has been loaded correctly
dataset


# In[3]:


# Set the pandas option to display all columns when printing a DataFrame
# By default, pandas may truncate columns when printing large DataFrames
# Setting max_columns to None ensures that all columns are displayed

pd.options.display.max_columns=None


# In[4]:


# Display the first 5 rows of the dataset
# This is useful to quickly inspect the structure of the dataset, 
# including column names and data types, and to get a sense of the data

dataset.head()


# In[5]:


# Display the shape of the dataset as (number of rows, number of columns)
# This is useful to understand the dimensions of the dataset
# It returns a tuple: (number of rows, number of columns)

dataset.shape


# In[6]:


# Display summary information about the dataset, including the number of entries, column names, 
# data types, and the number of non-null values for each column.

dataset.info()


# In[7]:


# Calculate and display the number of missing (null) values for each column in the dataset.

dataset.isnull().sum()


# In[8]:


# Count the number of duplicate rows in the dataset.

dataset.duplicated().sum()


# In[9]:


# Remove duplicate rows from the dataset to ensure each entry is unique.

dataset.drop_duplicates(inplace=True)


# In[10]:


# Verify if there are any remaining duplicate rows in the dataset after removing duplicates.

dataset.duplicated().sum()


# In[11]:


# Generate descriptive statistics for the dataset, including measures such as count, mean,
# standard deviation, min, max, and quartiles for numerical columns.

dataset.describe()


# In[12]:


# Count the number of occurrences of each unique value in the 'Class' column.

dataset['Class'].value_counts()


# In[13]:


# Initialize the StandardScaler and apply it to standardize the 'Amount' column in the dataset.

sc = StandardScaler()
dataset['Amount'] = sc.fit_transform(pd.DataFrame(dataset['Amount']))


# In[14]:


# Display the first few rows of the dataset to inspect the initial data.

dataset.head()


# In[15]:


# Convert the 'Time' column to datetime format for proper time series analysis.

dataset['Time'] = pd.to_datetime(dataset['Time'])


# In[16]:


# Display updated summary information about the dataset, including data types and non-null counts after recent transformations.

dataset.info()


# In[17]:


# Compute the correlation matrix for the dataset to understand the relationships between numerical variables.

cor = dataset.corr()
cor


# In[18]:


# Create a heatmap to visualize the correlation matrix of the dataset with annotations and a color map.

plt.figure(figsize=(24,18))
sns.heatmap(cor, cmap="cividis", annot=True)
plt.title("Correlation Matrix",fontsize=16)
plt.savefig("correlation_matrix.png")
plt.show()


# In[19]:


# Create a count plot for the 'Class' column to visualize the distribution of different class labels in the dataset.

sns.countplot(x='Class', data=dataset, palette=['skyblue', 'salmon'])

# Add value annotations to each bar in the count plot
for p in plt.gca().patches:
    plt.gca().annotate(format(p.get_height(), 'd'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
plt.title("Count of Fraud vs Non-Fraud Transactions")
plt.savefig("Transaction_Counts.png")
plt.show()


# In[20]:


# Distribution of 'Amount' feature (before scaling)

plt.figure(figsize=(10, 6))
sns.histplot(dataset['Amount'], bins=50, kde=True)
plt.title("Distribution of Transaction Amount", fontsize=16)
plt.savefig("Transaction_Distribution.png")
plt.show()


# In[21]:


# Pairplot of selected features for Fraud vs Non-Fraud (reduced sample for better visualization)

subset = dataset.sample(frac=0.1, random_state=42)
sns.pairplot(subset, hue='Class', vars=['V1', 'V2', 'V3', 'Amount'], palette = 'Set1')
plt.title("Pairplot of Selected Features for Fraud and Non-Fraud")
plt.savefig("Pairplot.png")
plt.show()


# In[22]:


# Pie chart to visualize the class distribution (Fraud vs Non-Fraud)
classes = dataset['Class'].value_counts()
normal_share = classes[0] / dataset['Class'].count() * 100
fraud_share = classes[1] / dataset['Class'].count() * 100
labels = ['Non-Fraudulent', 'Fraudulent']
sizes = [normal_share, fraud_share]
explode = (0, 0.1)
colors = ['#9467bd', '#fdae61']  # Purple and Yellow
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90,colors=colors)
plt.title("Fraud vs Non-Fraud Transactions Share")
plt.axis('equal')
plt.savefig("class_distribution.png")
plt.show()


# In[23]:


# Count the number of occurrences of each unique value in the 'Class' column to understand the distribution of class labels.

dataset['Class'].value_counts()


# In[24]:


# Calculate the percentage share of normal and fraudulent transactions.
# `normal_share` represents the percentage of non-fraudulent transactions.
# `fraud_share` represents the percentage of fraudulent transactions.

classes = dataset['Class'].value_counts()
normal_share = classes[0] / dataset['Class'].count() * 100
fraud_share = classes[1] / dataset['Class'].count() * 100


# In[25]:


# Define labels for each segment of the pie chart and their respective sizes.
# `explode` determines the fraction of the radius to offset each slice.
# `fig1` and `ax1` create a subplot for the pie chart.
# `ax1.pie()` generates the pie chart with specified parameters including segment labels, sizes, and formatting.
# `ax1.axis('equal')` ensures the pie chart is circular.

labels = 'Non-Fraudulent', 'Fraudulent'
sizes = [normal_share, fraud_share]
explode = (0, 0.1)
colors = ['#e377c2', '#1f77b4']  # Pink and Blue
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Fraud vs Non-Fraud Transactions Share class distribution")
plt.savefig("Share_class_distribution.png")
plt.show()


# In[26]:


# Drop the 'Time' column from the dataset as it is no longer needed for analysis.

dataset = dataset.drop(['Time'], axis=1)


# In[27]:


# Display the first few rows of the dataset to preview the current state after previous modifications.

dataset.head()


# In[28]:


# Separate the features and the target variable from the dataset.
# `X` contains all columns except 'Class', which will be used as features.
# `y` contains the 'Class' column, which will be used as the target variable for prediction.

X = dataset.drop('Class', axis=1)
y = dataset['Class']


# In[29]:


# Display the shape of the feature matrix `X` to understand its dimensions, 
# i.e., the number of samples and the number of features.

X.shape


# In[30]:


# Display the shape of the target vector `y` to understand its dimensions, i.e., the number of samples.

y.shape


# In[31]:


# Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
# `X_res` and `y_res` contain the resampled feature matrix and target vector, respectively,
# with the minority class over-sampled to balance the class distribution.

X_res, y_res = SMOTE().fit_resample(X, y)


# In[32]:


# Display the value counts of the resampled target vector `y_res` to check the distribution of the classes
# after applying SMOTE and ensuring that the classes are balanced.

y_res.value_counts()


# In[33]:


# Split the resampled dataset into training and testing sets.
# `X_train` and `y_train` are used for training the model.
# `X_test` and `y_test` are used for evaluating the model's performance.
# `test_size=0.2` specifies that 20% of the data will be used for testing.
# `random_state=42` ensures reproducibility of the split.

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[34]:


# Display the number of samples in the training set `X_train` to understand the size of the training data.

len(X_train)


# In[35]:


# Display the number of samples in the testing set `X_test` to understand the size of the testing data.

len(X_test)


# In[36]:


# Initialize a logistic regression model.
# Train the model on the training data `X_train` and `y_train`.

model = LogisticRegression()
model.fit(X_train, y_train)


# In[37]:


# Use the trained logistic regression model to make predictions on the testing data `X_test`.
# `y_pred` contains the predicted class labels for the test set.

y_pred = model.predict(X_test)


# In[38]:


# Display the predicted class labels for the test set `X_test` to review the model's predictions.

y_pred


# In[39]:


# Calculate evaluation metrics for the logistic regression model.
# `accuracy` measures the overall correctness of the model's predictions.
# `precision` measures the model's ability to correctly identify positive instances, weighted by support.
# `recall` measures the model's ability to find all positive instances, weighted by support.
# `f1` is the harmonic mean of precision and recall, providing a single score to evaluate model performance.

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


# In[40]:


# Print the evaluation metrics for the logistic regression model.
# Display the accuracy, precision, recall, and F1 score to assess model performance.

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[41]:


# Initialize a decision tree classifier with a fixed random state for reproducibility.
# Train the decision tree model on the training data `X_train` and `y_train`.

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


# In[42]:


# Use the trained decision tree classifier to make predictions on the testing data `X_test`.
# `y_pred_dt` contains the predicted class labels for the test set.

y_pred_dt = dt.predict(X_test)
y_pred_dt


# In[43]:


# Calculate evaluation metrics for the decision tree classifier.
# `accuracy` measures the overall correctness of the model's predictions.
# `precision` measures the model's ability to correctly identify positive instances, weighted by support.
# `recall` measures the model's ability to find all positive instances, weighted by support.
# `f1` is the harmonic mean of precision and recall, providing a single score to evaluate model performance.

accuracy = accuracy_score(y_test, y_pred_dt)
precision = precision_score(y_test, y_pred_dt, average='weighted')
recall = recall_score(y_test, y_pred_dt, average='weighted')
f1 = f1_score(y_test, y_pred_dt, average='weighted')


# In[44]:


# Print the evaluation metrics for the decision tree classifier.
# Display the accuracy, precision, recall, and F1 score to assess model performance.

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[45]:


# Initialize a random forest classifier with a specified number of trees (n_estimators=5) 
# and a fixed random state for reproducibility.

classifier = RandomForestClassifier(n_estimators=5, random_state=42)


# In[46]:


# Train the random forest classifier on the training data `X_train` and `y_train`.

classifier.fit(X_train, y_train)


# In[47]:


# Use the trained random forest classifier to make predictions on the testing data `X_test`.
# `y_pred_rf` contains the predicted class labels for the test set.

y_pred_rf = classifier.predict(X_test)
y_pred_rf


# In[48]:


# Calculate evaluation metrics for the random forest classifier.
# `accuracy` measures the overall correctness of the model's predictions.
# `precision` measures the model's ability to correctly identify positive instances, weighted by support.
# `recall` measures the model's ability to find all positive instances, weighted by support.
# `f1` is the harmonic mean of precision and recall, providing a single score to evaluate model performance.

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')


# In[49]:


# Print the evaluation metrics for the random forest classifier.
# Display the accuracy, precision, recall, and F1 score to assess model performance.

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[50]:


# Initialize an XGBoost classifier with a fixed random state for reproducibility.

xgb_model = xgb.XGBClassifier(random_state=42)


# In[51]:


# Train the XGBoost classifier on the training data `X_train` and `y_train`.

xgb_model.fit(X_train, y_train)


# In[52]:


# Use the trained XGBoost classifier to make predictions on the testing data `X_test`.
# `y_pred_xgb` contains the predicted class labels for the test set.

y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb


# In[53]:


# Calculate evaluation metrics for the XGBoost classifier.
# `accuracy` measures the overall correctness of the model's predictions.
# `precision` measures the model's ability to correctly identify positive instances, weighted by support.
# `recall` measures the model's ability to find all positive instances, weighted by support.
# `f1` is the harmonic mean of precision and recall, providing a single score to evaluate model performance.

accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb, average='weighted')
recall = recall_score(y_test, y_pred_xgb, average='weighted')
f1 = f1_score(y_test, y_pred_xgb, average='weighted')


# In[54]:


# Print the evaluation metrics for the XGBoost classifier.
# Display the accuracy, precision, recall, and F1 score to assess model performance.

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[55]:


# Calculate the confusion matrix for the logistic regression model.
# `cm` contains the confusion matrix values.
# `ConfusionMatrixDisplay` is used to visualize the confusion matrix with appropriate labels.

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='inferno')
plt.title("Confusion Matrix for Logistic Regression")
plt.savefig("confusion_matrix_lr.png")
plt.show()


# In[56]:


# Compute the ROC curve for the logistic regression model.
# `fpr` and `tpr` represent the false positive rates and true positive rates, respectively.
# `thresholds` are the decision thresholds used to compute the ROC curve.
# `roc_auc` calculates the area under the ROC curve, a measure of the model's performance.

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Create a RocCurveDisplay object to visualize the ROC curve with the computed fpr, tpr, and roc_auc.
# `estimator_name` specifies the name of the model for display purposes.

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='LogisticRegression')

# Plot the ROC curve and set the title for the plot.
display.plot()
plt.title("Plot of ROC Curve for Logistic Regression Model")
plt.savefig("roc_curve_lr.png")
plt.show()


# In[57]:


# Calculate the confusion matrix for the decision tree classifier.
# `cm` contains the confusion matrix values.
# `ConfusionMatrixDisplay` is used to visualize the confusion matrix with appropriate labels.

cm = confusion_matrix(y_test, y_pred_dt, labels=dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot(cmap='Oranges')
plt.title("Confusion Matrix for Decision Tree Classifier ")
plt.savefig("confusion_matrix_dtc.png")
plt.show()


# In[58]:


# Compute the ROC curve for the decision tree classifier.
# `fpr` and `tpr` represent the false positive rates and true positive rates, respectively.
# `thresholds` are the decision thresholds used to compute the ROC curve.
# `roc_auc` calculates the area under the ROC curve, a measure of the model's performance.

fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt)
roc_auc = auc(fpr, tpr)

# Create a RocCurveDisplay object to visualize the ROC curve with the computed fpr, tpr, and roc_auc.
# `estimator_name` specifies the name of the model for display purposes.

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='DecisionTreeClassifier')

# Plot the ROC curve and set the title for the plot.
display.plot()
plt.title("Plot of ROC Curve for Decision Tree Model")
plt.savefig("roc_curve_dtc.png")
plt.show()


# In[60]:


# Calculate the confusion matrix for the random forest classifier.
# `cm` contains the confusion matrix values.
# `ConfusionMatrixDisplay` is used to visualize the confusion matrix with appropriate labels.

cm = confusion_matrix(y_test, y_pred_rf, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap='cubehelix')
plt.title("Confusion Matrix for Random Forest Classifier ")
plt.savefig("confusion_matrix_rfc.png")
plt.show()


# In[61]:


# Compute the ROC curve for the random forest classifier.
# `fpr` and `tpr` represent the false positive rates and true positive rates, respectively.
# `thresholds` are the decision thresholds used to compute the ROC curve.
# `roc_auc` calculates the area under the ROC curve, a measure of the model's performance.

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr, tpr)

# Create a RocCurveDisplay object to visualize the ROC curve with the computed fpr, tpr, and roc_auc.
# `estimator_name` specifies the name of the model for display purposes.

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='RandomForestClassifier')

# Plot the ROC curve and set the title for the plot.
display.plot()
plt.title("Plot of ROC Curve for Random Forest Model")
plt.savefig("roc_curve_rfc.png")
plt.show()


# In[62]:


# Calculate the confusion matrix for the XGBoost classifier.
# `cm` contains the confusion matrix values.
# `ConfusionMatrixDisplay` is used to visualize the confusion matrix with appropriate labels.

cm = confusion_matrix(y_test, y_pred_xgb, labels=xgb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_model.classes_)

# Plot the confusion matrix and display it.
disp.plot(cmap = 'Greens')
plt.title("Confusion Matrix for XGBoost Classifier ")
plt.savefig("confusion_matrix_xgb.png")
plt.show()


# In[63]:


# Compute the ROC curve for the XGBoost classifier.
# `fpr` and `tpr` represent the false positive rates and true positive rates, respectively.
# `thresholds` are the decision thresholds used to compute the ROC curve.
# `roc_auc` calculates the area under the ROC curve, a measure of the model's performance.

fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb)
roc_auc = auc(fpr, tpr)

# Create a RocCurveDisplay object to visualize the ROC curve with the computed fpr, tpr, and roc_auc.
# `estimator_name` specifies the name of the model for display purposes.

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='xgb.XGBClassifier')

# Plot the ROC curve and set the title for the plot.
display.plot()
plt.title("Plot of ROC Curve for XGB Model")
plt.savefig("roc_curve_xgb.png")
plt.show()


# In[64]:


# Create an empty dataframe to store the scores for various algorithms.
# The dataframe will have columns for model name, accuracy score, precision score, recall score, AUC score, and f1 score.

score_card = pd.DataFrame(columns=['model_name', 'Accuracy Score', 'Precision Score', 'Recall Score', 'AUC Score', 'f1 Score'])

# Function to update the score card with performance metrics for a given model.
# `y_test` is the true labels, `y_pred` is the predicted labels, and `model_name` is the name of the model.

def update_score_card(y_test, y_pred, model_name):
    
    # Assign 'score_card' as a global variable to update the dataframe defined outside the function.
    global score_card
    
    # Append the results to the dataframe 'score_card'.
    # 'ignore_index=True' ensures the index is not considered when concatenating.
    score_card = pd.concat([
        score_card,
        pd.DataFrame([{
            'model_name': model_name,
            'Accuracy Score': accuracy_score(y_test, y_pred),
            'Precision Score': precision_score(y_test, y_pred),
            'Recall Score': recall_score(y_test, y_pred),
            'AUC Score': roc_auc_score(y_test, y_pred),
            'f1 Score': f1_score(y_test, y_pred)
        }])
    ], ignore_index=True)


# In[65]:


# Update the score card with performance metrics for each model.
# `update_score_card` function is called with test labels, predicted labels, and model names.

update_score_card(y_test, y_pred, 'Logistic Regression')
update_score_card(y_test, y_pred_dt, 'Decision Tree')
update_score_card(y_test, y_pred_rf, 'Random Forest')
update_score_card(y_test, y_pred_xgb, 'XG Boost classifier')

# Display the score card dataframe containing the performance metrics for all models.
score_card


# In[66]:


# Initialize an XGBoost classifier with a specified random state for reproducibility.
# `xgb.XGBClassifier` is used to create the model instance.

xgb_model = xgb.XGBClassifier(random_state=42)


# In[67]:


# Define the parameter grid for hyperparameter tuning using GridSearchCV.
# `param_grid` includes different values for the number of estimators, the maximum number of features to consider, and the criterion for splitting nodes.

param_grid = {
    'n_estimators': [100, 500],          # Number of trees in the forest.
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split.
    'criterion': ['gini', 'entropy']     # Function to measure the quality of a split.
}


# In[68]:


# Initialize GridSearchCV to perform hyperparameter tuning for the XGBoost classifier.
# `param_grid` defines the parameters to be searched.
# `cv=5` specifies 5-fold cross-validation.

gridSearch = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data to find the best parameters.
gridSearch.fit(X_train, y_train)


# In[69]:


# Print the best hyperparameters found by GridSearchCV.
# `gridSearch.best_params_` contains the best parameter values for the model.
# `gridSearch.best_score_` gives the best cross-validation accuracy achieved with those parameters.

print("Best Parameters:", gridSearch.best_params_)
print("Best Cross-validation Accuracy:", gridSearch.best_score_)


# In[70]:


# Retrieve the best model from the GridSearchCV object.
# `gridSearch.best_estimator_` provides the model with the best hyperparameters.

best_model = gridSearch.best_estimator_

# Predict the labels for the test set using the best model.
# `best_model.predict(X_test)` returns the predicted labels for the test data.

y_pred_best = best_model.predict(X_test)


# In[71]:


# Print the classification report for the predictions made by the best model.
# The report includes precision, recall, f1-score, and support for each class.

print("Classification Report:")
print(classification_report(y_test, y_pred_best))


# In[72]:


# Define the parameter distributions for hyperparameter tuning using RandomizedSearchCV.
# `param_distributions` includes various values for regularization strength, penalty types, solvers, iteration limits, 
# and class weights.

param_distributions = {
    'C': [0.001, 0.01, 0.1, 1],                      # Regularization strength.
    'penalty': ['l1', 'l2'],                         # Penalty type.
    'solver': ['liblinear', 'saga'],                 # Solver to use.
    'max_iter': [100, 200],                          # maximum number of iterations.
    'class_weight': [None, 'balanced']               # Class weights
}

# Setup RandomizedSearchCV to find the best hyperparameters using random sampling.
# `n_iter=20` specifies the number of different combinations to try.
# `cv=5` sets up 5-fold cross-validation.
# `scoring='roc_auc'` optimizes the ROC AUC score.
# `n_jobs=-1` uses all available processors.
# `verbose=2` provides detailed output during the search.

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,                      
    cv=5,                        
    scoring='roc_auc',             
    n_jobs=-1,                     
    verbose=2,                   
    random_state=42              
)

# Fit the RandomizedSearchCV to the training data.
# This will search for the best hyperparameters using the specified parameter distributions.
random_search.fit(X_train, y_train)

# Retrieve the best model from RandomizedSearchCV.
# `best_model_lr` contains the model with the optimal hyperparameters found during the search.
best_model_lr = random_search.best_estimator_

# Predict probabilities for the test set using the best model.
# `predict_proba` provides probability estimates for each class, `[:, 1]` selects the probabilities for the positive class.
y_pred_lr = best_model_lr.predict_proba(X_test)[:, 1]

# Calculate and print the ROC AUC score for the best model on the test set.
# `roc_auc_score` evaluates the model's performance in distinguishing between classes.
roc_auc = roc_auc_score(y_test, y_pred_lr)
print(f'Best ROC AUC Score: {roc_auc:.4f}')

# Print the best parameters found by RandomizedSearchCV.
print(f'Best Parameters: {random_search.best_params_}')


# In[73]:


# Predict the labels for the test set using the best logistic regression model found by RandomizedSearchCV.
# `predict` returns the predicted labels for the test data.

y_pred_lr = best_model_lr.predict(X_test)

# Print the classification report for the predictions made by the best model.
# The report includes precision, recall, f1-score, and support for each class.

print("Classification Report:")
print(classification_report(y_test, y_pred_lr))


# In[74]:


# Define the Stratified K-Fold cross-validator.
# `n_splits=5` specifies 5 folds for cross-validation.
# `shuffle=True` shuffles the data before splitting to ensure randomness.
# `random_state=42` ensures reproducibility.

strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data using the best model.
# `cross_val_score` computes the ROC AUC score for each fold of the cross-validation.
# `cv=strat_k_fold` uses the previously defined Stratified K-Fold cross-validator.
# `scoring='roc_auc'` evaluates the ROC AUC score.
# `n_jobs=-1` uses all available processors for computation.

cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=strat_k_fold, scoring='roc_auc', n_jobs=-1)

# Print the ROC AUC scores for each fold in the cross-validation.
print(f'Cross-Validation ROC AUC Scores: {cross_val_scores}')

# Print the mean ROC AUC score across all folds.
print(f'Mean ROC AUC Score: {cross_val_scores.mean():.4f}')

# Print the standard deviation of the ROC AUC scores across all folds.
print(f'Standard Deviation of ROC AUC Scores: {cross_val_scores.std():.4f}')


# In[75]:


# Setup GridSearchCV to perform hyperparameter tuning for the XGBoost model.
# `param_grid` specifies the parameters to search over.
# `cv=5` sets up 5-fold cross-validation for evaluating each set of parameters.

Model = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV to the resampled training data.
# This process involves training multiple models with different parameter combinations and selecting the best one based on cross-validation results.

Model.fit(X_res, y_res)


# In[76]:


# Import the joblib library.
# Joblib is used for serializing and deserializing Python objects, 
# which is useful for saving and loading machine learning models.

import joblib


# In[77]:


# Save the trained GridSearchCV model to a file using joblib.
# This allows the model to be reused later without needing to retrain it.
# The model is saved with the filename "Credit_card_fraud_detection_model".

joblib.dump(Model, "Credit_card_fraud_detection_model")


# In[78]:


# Load the previously saved GridSearchCV model from the file using joblib.
# This allows you to use the model for predictions or further evaluation without retraining it.

model1 = joblib.load("Credit_card_fraud_detection_model")


# In[79]:


# Use the loaded model to make predictions on new data.
# The input data is a single instance with feature values provided in a list.
# `predict` will output the predicted class for this instance.

pred1 = model1.predict([[-1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 
                         0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, 
                         -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 
                         0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 
                         0.128539, -0.189115, 0.133558, -0.021053, 0.244200]])


# In[80]:


# Check the predicted class from the model.
# If the predicted class is 0, it indicates a non-fraudulent transaction.
# Otherwise, it indicates a fraudulent transaction.

if pred1 == 0:
    print("Non Fraudulent Transaction")
else:
    print("Fraudulent Transaction")


# In[ ]:




