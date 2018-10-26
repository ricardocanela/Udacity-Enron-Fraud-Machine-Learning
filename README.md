# Udacity-Enron-Fraud-Machine-Learning

Using Machine Learning to Identify Fraud in the Enron Corpus

### File Information

poi_id.py: Main file. Runs final feature selection, feature scaling, various classifiers (optional) and their results. Finally, dumps classifier, dataset and feature list so anyone can check results.  
tester.py: Functions for validation and evaluation of classifier, dumping and loading of pickle files.  
my_classifier.pkl: Pickle file for final classifier from poi_id.py.  
my_dataset.pkl: Pickle file for final dataset from poi_id.py.  
my_feature_list.pkl: Pickle file for final feature list from poi_id.py.  

/tools/:  

feature_format.py: Functions to convert data from dictionary format into numpy arrays and separate target label from features to make it suitable for machine learning processes.  
feature_creation.py: Functions for creating two new features - 'poi_email_ratio' and 'exercised_stock_options'.  
select_k_best.py: Function for selecting k best features using Sci-kit Learn's SelectKBest, sorts them in descending order of score.  

### Abstract
Enron's financial scandal in 2001 led to the creation of a very valuable dataset for machine learning, one where algorithms were trained and tested to be able to find fraudulent employees, or persons-of-interest (POIs). In this project, a merged dataset of financial and email data will be used to go through the entire machine learning process.  

### Project Goal
The aim of this project is to apply machine learning techniques to build a predictive model that identifies Enron employees that may have committed fraud based on their financial and email data.

The boffins at Udacity have combined important aspects of the financial and email data to create one consolidated dataset that will be used for creating the predictive model. The dataset has 14 financial features (salary, bonus, etc.), 6 email features (to and from messages, etc.) and a boolean label that denotes whether a person is a person-of-interest (POI) or not (established from credible news sources). It is these features that will be explored, cleaned, and then put through various machine learning algorithms, before finally tuning them and checking its accuracy (precision and recall). The objective is to get a precision and recall score of at least 0.3.

### Why Machine Learning?
Machine Learning is an incredibly effective field when it comes to making predictions from data, especially large amounts of it. The Enron Corpus, after cleaning, has around 500,000 emails, and attempting to identify fraudulent employees by manually foraging through half a million emails is a daunting task at best. Designing and implementing a machine learning algorithm can significantly reduce the legwork, although making sure the algorithm is producing accurate results is challenging but ultimately worth the effort.

### Exploring the Dataset
explore_enron_data.py has the code that explores the dataset and provides the following information.

The dataset is a dictionary of dictionaries, with the keys being 146 Enron employees, and their values being a dictionary of features. The features available for each person are:
```
['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person']
```
In the features, the boolean poi denotes whether the person is a person of interest or not. The Poi_count function shows that there are 18 such people in the dataset, and the aim of this project is to find distinguishing features that set these people apart from the others.

I have plotted the number of missing values for each feature, here is the result:

![missing_values](https://i.imgur.com/rYiMMLK.png)

### Outliers
Ofcourse, not everyone has data for each feature, and missing data is denoted by 'NaN'. The NaN_count function prints a dictionary sorted in descending order. Using the results of the function, as well as going through the names, I came across the following outliers:

LOCKHART EUGENE E: No data available on this person.  
THE TRAVEL AGENCY IN THE PARK: Not a person/employee associated with Enron.  
TOTAL: Summation of everyone's data - likely part of a spreadsheet. Found after visualizing financial features and finding an extreme outlier.  
I will remove these outliers (and any others I find) after I complete feature selection.  

### Feature Selection

Select K Best and Feature Creation
There are lots of features available to play with, but as with all lists of features, not all of them are useful in predicting the target variable. My first step is to check the ability of each feature in clearly differentiating between POI and non-POI. To do this, I'm going to use Scikit-Learn's SelectKBest algorithm, which will give me a score for each feature in it's ability to identify the target variable.

In select_k_best.py, the Select_K_Best function returns an array of k tuples in descending order of its score. Running this will show the most useful features and the not-so-useful ones.

To know the best value of K to use in Select_K_Best, I have plotted a pipeline which runs a DecisionTree with SelectKBest using a GridSearchCV for all 19 present features. The result is that the best K is 4. Here is the plot:

![k_best](https://i.imgur.com/l1Ikrh3.png)

And running the Select_K_Best with K equals 4, these are the selected features:

```
['shared_receipt_with_poi', 
'from_poi_to_this_person', 
'loan_advances', 
'from_this_person_to_poi']
```

#### Creating The Features

In `feature_creation.py`, the `CreatePoiEmailRatio` function creates the `'poi_email_ratio'` feature and the `CreateExercisedStockRatio` function creates the `'exercised_stock_ratio'` feature. Testing these features on a few people (`test_people` in the file), however, led to a few complications, namely with the `CreatePoiEmailRatio` function.

As explained above, if all a person's emails sent/received were to/from POIs, then the value should be 2 (i.e. the maximum value). However, Mark Frevert has a value of 11.5, and Kenneth Lay has a value of 3.4, above the designed maximum. Checking the message totals for these two people (see `explore_enron_data.py`), I noticed that the `'from_poi_to_this_person'` total exceeded the `'from_messages'` total (and hence it must be possible that the `'from_this_person_to_poi'` total can exceed the `'to_messages'` total). This led me to conclude that the from and to message totals must not include the emails that are sent to/received from POIs, and I'll have to recalibrate the function to add them.

After fixing the error, I ran the `Select_K_Best` function again, and `'poi_email_ratio'` came 5th with a score of 16.2, but unfortunately `'exercised_stock_ratio'` didn't work at all, coming dead last with a score nearly twice as bad as the one before it.

#### Comparing difference with and without created features

To see if the created features has made good changes in our prediction model, a DecisionTree classifier was tested with the 4 original features and then with the new `'poi_email_ratio'` feature, this is the result:


| Model                    | Precision | F1 |
| :----------------------- | --------: | -----: |
| DecisionTree             | 0.147     | 0.170  |
| DecisionTree + new feat  | 0.312     | 0.267  |


#### Final Feature Selection

The final 5 features to be used for the machine learning process are:

```
['shared_receipt_with_poi', 
'from_poi_to_this_person', 
'loan_advances', 
'from_this_person_to_poi',
'poi_email_ratio']
```

#### Feature Scaling

A lot of the features have vastly different scales. Financial features such as salary range from below ten thousand up to just above a million dollars, whereas email features like to and from messages max out at around a few thousand. The newly created `'poi_email_ratio'` has a maximum of two. To make sure the scales are comparable, I'm going to run sklearn's `MinMaxScaler()` over all the features.


### Selecting An Algorithm

The next step in the process is to try out a number of algorithms to see which ones do a good job to correctly predicting (or not making an error with) POIs. The algorithms I'm going to try out are:

* `DecisionTreeClassifier`: A standard decision tree classifier
* `AdaBoost`: An ensemble decision tree classifier
* `K-Nearest Neighbors`: A proximity based classifier

#### Decision Tree Classifier

Here are the results of using the `DecisionTreeClassifier` while changing the `min_samples_split` parameter over the  feature list (5 features):

| Min Samples Split | Precision | Recall |
| :---------------- | --------: | -----: |
| 10                | 0.273     | 0.204  |
| 5                 | 0.294     | 0.266  |
| 4                 | 0.294     | 0.270  |
| 3                 | 0.296     | 0.274  |
| 2                 | 0.288     | 0.283  |

obs: Its important to make tuning in the parameters and compare the diferent results to maximize your predictor score, since each predictor model has its own hyperparameters and changing them may affect drastically the precision of your model.

##### With smaller feature lists

Since I got better results when reducing the number of features, I thought I'd go a little further. I decided to try just 2 features: the best financial feature, `'exercised_stock_options'`, and the best email feature, `'poi_email_ratio'`. These are the results changing `min_samples_split`:

| Min Samples Split | Precision | Recall | F1 Score |
| :---------------- | --------: | -----: | -------: |
| 2                 | 0.505     | 0.465  | 0.485    |
| 3                 | 0.586     | 0.460  | 0.515    |
| 4                 | 0.596     | 0.462  | 0.521    |
| 5                 | 0.601     | 0.461  | 0.520    |
| 6                 | 0.623     | 0.456  | 0.526    |
| 7                 | 0.654     | 0.455  | 0.536    |
| 8                 | 0.654     | 0.452  | 0.534    |
| 9                 | 0.659     | 0.451  | 0.535    |
| 10                | 0.673     | 0.442  | 0.534    |
| 15                | 0.742     | 0.414  | 0.531    |         
| 20                | 0.714     | 0.312  | 0.434    |  


#### AdaBoost

The Adaboost Classifier is an ensemble decision tree classifier that repeatedly iterates estimators over the dataset.

Results with full feature list changing number of estimators - `n_estimators`:

| No. of Estimators | Precision | Recall |
| :---------------- | --------: | -----: |
| 10                | 0.356     | 0.227  |
| 25                | 0.354     | 0.268  |
| 50                | 0.404     | 0.306  |
| 75                | 0.446     | 0.317  |
| 100               | 0.461     | 0.315  |


These results show that the Adaboost Classifier performs better than the Decision Tree Classifier, but it is much, much more computationally expensive, with the 100 estimators taking a few minutes to run.

#### K-Nearest Neighbors

Results while changing the number of nearest neighbors:

| No. of Neighbors  | Precision | Recall |
| :---------------- | --------: | -----: |
| 1                 | 0.157     | 0.131  |
| 2                 | 0.276     | 0.066  |
| 3                 | 0.503     | 0.245  |
| 4                 | 0.590     | 0.107  |
| 5                 | 0.639     | 0.168  |
| 6                 | 1.000     | 0.004  |
| 7                 | 0.897     | 0.018  |

K-Nearest neighbors gives incredibly precise predictions as the value increases past 5, but at the cost of recall. The reason for the near-perfect precision is that it makes very few POI predictions in the first place, but when it does, it's accurate. For example, for 6 neighbors, just 8 POI predictions were made, with all 8 being true positives. While this precision seems compelling, it comes at the cost of recall, which is astoundingly low because when one makes very few POI predictions (that is, a lot of the predictions are that each person is *not* a POI), one lets a lot of actual POIs through. For this reason, K-Nearest Neighbors is not an appropriate classifier.


### Final Algorithm

The algorithm I've chosen is a Decision Tree Classifier with `'min_samples_split'` value of 8 and a feature list of `['poi', 'exercised_stock_options', 'poi_email_ratio']`.

The feature list looks a bit empty considering there are so many to play with, but through experimentation I've noticed this produces the best precision and recall scores. The visualization of the two features also produces a less overplotted graph, which is good for decision tree classifiers (see below).

These are the evaluation metric results from `tester.py`:

| Evaluation Metric | Score |
| :---------------- | -----:|
| Accuracy          | 0.869 |
| Precision         | 0.654 |
| Recall            | 0.455 |
| F1                | 0.536 |
| F2                | 0.483 |
| Total Predictions | 12000 |
| True Positives    |   907 |
| False Positives   |   483 |
| True Negatives    |  9517 |
| False Negatives   |  1093 |


### Validation and Evaluation

#### Validation

Certain validation and evaluation metrics are used to get the algorithm performance scores above, and now I'm going to go into a little more detail on what they are and why they're important or right for this data.

The first validation step is to separate the dataset into training and testing sets. Why is this important? If an algorithm learns from and tests on the same data, then the model would just repeat the target labels and likely get a perfect score. The algorithm will *overfit* to the particular data, and should it be tested any new, unseen data (say another company's data after similar circumstances), it will likely perform very poorly in its predictions.

To perform on independent datasets and combat overfitting, sklearn has several helper functions under Cross-validation. The one used in this project is `StratifiedShuffleSplit`, which is particularly suitable for this dataset. Since there is a large imbalance in the number of POIs and non-POIs (very few POIs in entire dataset), it's possible that in a random split that there won't be a lot of POI labels to either train or validate on. What `StratifiedShuffleSplit` does is that it ensures that the percentage of target labels is approximately the same in both training and validation sets as it is in the complete dataset.


#### Evaluation

The most common measure of an algorithm's performance is its accuracy score - the number of predicted labels it got correct over all predicted labels. We'll see, however, that this isn't a suitable metric for this kind of dataset. When there's a large imbalance in the data labels, then a high accuracy score can be achieved by simply predicting all labels to be the most common label in the dataset. In this context, it would be predicting all labels to be non-POI, which will still result in an accuracy score above 80% despite not actually predicting anything.

In these situations, a much better evaluation metric for the algorithm is the precision score and the recall score (or the F1 score, which is a weighted average of the two). The precision score is the ratio of the true positives over the sum of true and false positives, while the recall score is the ratio of the true positives over the sum of true positives and false negatives. In the context of Enron data, the precision score measures the algorithm's ability to, given that it is *predicting* a POI, correctly predicts a POI, while the recall score measures the algorithm's ability to, given that it *is* a POI, correctly predicts a POI. The difference will be clearer once the true positives, false positives, true negatives and false negatives are defined in this context:

* **True positive**: when a label is a POI and the algorithm predicts it is a POI.
* **False positive**: when a label *is not* a POI and the algorithm predicts it is a POI.
* **True negative**: when a label is not a POI and the algorithm predicts a non-POI.
* **False negative**: when a label is not a POI and the algorithm predicts a POI.

So, the the precision score is how often the algorithm is getting the prediction of POI right, whereas the recall score is, given that the label is a POI, how often the algorithm predicts it is.


### Discussion

For feature scaling, I applied a `MinMaxScaler` to all the features, but towards the end of the project I wondered if there are other methods that I could have used. A `MinMaxScaler` just shrinks or expands the data along a scale, and it's useful when the two axes scales differ by a huge amount, but I passed the opportunity to transform the data into different shapes. Particularly, if I could transform the data to make the distribution Gaussian and see what the data looked like when normally distributed. I suspect the algorithms would work differently, and would lead to a lot more options (at the very least, it would lead to interesting outcomes).

The other aspect that was not touched in this project, primarily because it's beyond the scope of the analysis, was actually analyzing the text in the emails. The text learning mini-project provided an entry point into the world of natural language processing, but I suspect that using that data to help predict POIs is a much more challenging task. Nonetheless, it is interesting to think about, and hopefully soon there will be a stage where I'm skilled enough in machine learning to include that realm into my investigation.
