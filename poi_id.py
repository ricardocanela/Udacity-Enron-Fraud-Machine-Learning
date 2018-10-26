#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from feature_creation import CreatePoiEmailRatio, CreateExercisedStockRatio

sys.path.append("../final_project/")
from tester import dump_classifier_and_data, test_classifier

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.cross_validation import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

### Task 1: Select the features I'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                    'director_fees', 'to_messages', 'from_poi_to_this_person',
                    'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

import copy
all_features_list = copy.copy(features_list)

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Plotting GridSearchCV to get best K in SelectKBest

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

kbest = SelectKBest(f_classif)
pipeline = Pipeline([('kbest', kbest), ('clf', DecisionTreeClassifier(min_samples_split=7))])
all_k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
grid_search = GridSearchCV(pipeline, {'kbest__k': all_k})
grid_search.fit(features, labels)

scores = [x[1] for x in grid_search.grid_scores_]

plt.plot(all_k, scores)
plt.xlabel('Number of K')
plt.ylabel('Grid search score')
plt.show()

from select_k_best import Select_K_Best
features_from_k_best = Select_K_Best(data_dict, features_list, 4)
print features_from_k_best
# We found a good option to use 13 features using select_k_best, here are them

features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                      'salary', 'deferred_income', 'long_term_incentive',
                      'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                      'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']

features_list = ['poi','shared_receipt_with_poi', 'from_poi_to_this_person', 'loan_advances', 'from_this_person_to_poi']


# Checking missing values - Plotting a horizontal bar chart to show missing values

df = pd.DataFrame.from_dict(data_dict, orient='index')

missing_values = []
for feature in all_features_list[1:]:
    missing_values.append(sum(df[feature] == 'NaN'))

print len(df) # Size of dataframe is 146

print all_features_list
plt.barh(range(0,19),missing_values)
plt.yticks(range(0,19), all_features_list[1:])
plt.xlabel('Number of missing values')
plt.show()

### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Task 3: Create new feature(s)

##### Testing DecisionTreeClassifier with and without new features

clf = DecisionTreeClassifier(min_samples_split=7)

print 'without new features'
test_classifier(clf, data_dict, features_list, folds=1000)


### See feature_creation.py to see the code for the creation of the two features

CreatePoiEmailRatio(data_dict, features_list)
CreateExercisedStockRatio(data_dict, features_list)

##### Testing DecisionTreeClassifier with and without new features

print 'with new features'
test_classifier(clf, data_dict, features_list, folds=1000)

### After running Select_K_Best on the new features, poi_email_ratio comes 5th,
### but exercised_stock_ratio comes last. Let's remove it from the feature list.

features_list.remove('exercised_stock_ratio')

### Store to my_dataset for easy export.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Scale features

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


###### Decision Tree Classifier ######

clf = DecisionTreeClassifier(min_samples_split=7)

######################################


###### AdaBoost Classifier ######

# clf = AdaBoostClassifier(n_estimators=100)

#################################


###### K-Nearest Neighbors Classifier ######

# clf = KNeighborsClassifier(n_neighbors=6)

############################################



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print accuracy_score(labels_test, pred)
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)


### Trying out other feature lists, particularly for the DecisionTreeClassifier

DT_features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                 'salary', 'deferred_income', 'long_term_incentive',
                 'poi_email_ratio']

DT_features_list_email = ['poi', 'exercised_stock_options', 'poi_email_ratio']


### Proper testing of classifier

test_classifier(clf, my_dataset, DT_features_list_email, folds=1000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, DT_features_list_email)
