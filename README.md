# Identifying Fraud from Enron Email Project

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

The goal of this project is to analyse Enron Email data and develop a person of interest identifier based on financial and email data from the dataset. The dataset is quite large and contains multiple features. Therefore, it would be quite unfeasible to analyse all records manually. Machine Learning would be particularly useful for this task - I can build up a model and use it on a whole dataset to identify people of interest. However, the accuracy of using Machine Learning algorithms is rarely perfect, so not only I have to pick the right algorithm, but also tune it.

## Dataset
The dataset provided for training contains data on **146** Enron employees with **21** features for each employee. 14 of the features provide financial data, 6 of them provide email data and the last one specifies whether an employee is a person of interest or not.

However, not all employees have values for all features. Following snippet shows feature completeness (100% meaning that all values are present for a given feature):

```
salary: 65%
to_messages: 58%
deferral_payments: 26%
total_payments: 85%
loan_advances: 2%
bonus: 56%
email_address: 76%
restricted_stock_deferred: 12%
total_stock_value: 86%
shared_receipt_with_poi: 58%
long_term_incentive: 45%
exercised_stock_options: 69%
from_messages: 58%
other: 63%
from_poi_to_this_person: 58%
from_this_person_to_poi: 58%
poi: 100%
deferred_income: 33%
expenses: 65%
restricted_stock: 75%
director_fees: 11%
```

Features such as loan advances (2%) and director fees (11%) have very low occurrences in the dataset.

POI (person of interest) classification is present for all employees, but only **18** of them were identified as POIs. Having such a small number of POIs, I expect that it would be quite hard to reach high precision and recall scores for the classifier.

## Outliers

There were several outliers in the dataset, which were removed.

`TOTAL` - didn't represent an Enron employee, but rather provided sums of all feature values. Not particularly useful for the exploration

`THE TRAVEL AGENCY IN THE PARK` - didn't represent an Enron employee and didn't provide any useful information.


## Selected features
I have used SelectKBest to select best features. Following snippet demonstrates calculated scores:

```
exercised_stock_options: 24.82
total_stock_value: 24.18
bonus: 20.79
total_wealth_received: 19.75
salary: 18.29
from_this_person_to_poi_ratio: 16.41
deferred_income: 11.46
long_term_incentive: 9.92
restricted_stock: 9.21
total_payments: 8.77
shared_receipt_with_poi: 8.59
loan_advances: 7.18
expenses: 6.09
from_poi_to_this_person: 5.24
other: 4.19
from_poi_to_this_person_ratio: 3.13
from_this_person_to_poi: 2.38
director_fees: 2.13
to_messages: 1.65
deferral_payments: 0.22
from_messages: 0.17
restricted_stock_deferred: 0.07
```

I have also used MinMaxScaler to perform min-max scaling because it has slightly increased performance of my classifiers.

## Implemented features
Three new features have been implemented:

#### from_this_person_to_poi_ratio
Ratio of emails sent to a POI to a total number of emails sent.

When it comes to number of emails sent from an employee to a POI, absolute values might not represent an accurate picture. For example, if an employee sent 500 emails to a POI, we might suspect him as POI because the absolute value is high. However, if the same person sent over 100k emails to all emails, then suddenly 500 doesn't seem suspicious at all (i.e. HR staff). Therefore, relative number works much better in this case.

#### from_poi_to_this_person_ratio
Ratio of emails received by a POI to a total number of received emails.

Same rationale as in the case of `from_this_person_to_poi_ratio`.

#### total_wealth_received
Total wealth received by an Enron employee. Calculated as sum of the following feature values:
- salary
- deferred_income
- loan_advances
- bonus
- total_stock_value
- expenses
- other
- director_fees

Individual values might not look suspicious, but the POI might distribute wealth across multiple instruments. The intention of this feature is to catch such actions.


#### Effect on the final algorithm
No new features (baseline):
```
Precision: 0.4487
Recall: 0.28000
```


`total_wealth_received` implemented:
```
Precision: 0.46176
Recall: 0.32600
```
Noticeable increase in both precision and recall comparing to the baseline.


`from_this_person_to_poi_ratio` implemented:
```
Precision: 0.4202
Recall: 0.29250
```
Slight increase in recall and slight decrease in precision comparing to the baseline.

`from_poi_to_this_person` implemented:
```
Precision: 0.4488
Recall: 0.28050
```
Minimal increase in both recall and precision comparing to the baseline.

All new features implemented:
```
Precision: 0.45833
Recall: 0.34650
```
Noticeable increase in precision and significant increase in recall comparing to the baseline.

## Picking an algorithm
I have selected an algorithm that selects 9 K best features, then performs a Principal component analysis with 5 components and finally performs a Gaussian Naive Bayes. Following code demonstrates the pipeline:

```python
gaussian_clf = Pipeline(steps=[
    ("SKB", SelectKBest(f_classif, k=9)),
    ("PCA", PCA(whiten=True, n_components=5)),
    ("GaussianNB", GaussianNB())
])
```

I have also tried using logistic regression and adaptive boosting classifiers, but found their performance slightly lower comparing to my final selection. Following code demonstrates these classifiers:

```python
logistic_regression_clf = Pipeline(steps=[
    ("SelectKBest", SelectKBest(f_classif, k=10)),
    ("LogisticRegression", LogisticRegression(
        C=10**10,
        tol=10**-10,
        solver='liblinear'
    ))
])

adaboost_clf = Pipeline(steps=[
    ("SelectKBest", SelectKBest(f_classif, k=5)),
    ("AdaBoostClassifier", AdaBoostClassifier(
        algorithm='SAMME',
        n_estimators=50
    ))
])
```

In order to improve the performance of these algorithms, I have been tuning several parameters. For all algorithms I have tried different values of K best features. For the naive bayes pipeline, I have also tried different number of components and found that `5` works the best. For logistic regression and adaptive boosting classifiers I have tried different values of `C, tol, solver` and `algorithm, n_estimators` correspondingly.

Choosing ~10 features worked the best (naive bayes pipeline showed slightly better performance with 9 features). As soon as number of features was going higher or lower, the performance significantly dropped for all algorithms. Tuning number of components also made a noticeable difference for the naive bayes pipeline.

## Evaluation
I have used precision and recall as evaluation metrics. Both of these metrics are suitable for evaluating performance on a specific class, which is what I am doing in this project. According to the project's specification, both recall and evaluation should have values higher than 0.3 for the algorithm to be acceptable.

In the context of this project, precision rate indicates number of POIs that were identified correctly over all POIs identified (i.e. true positives + false positives). Recall indicates number of POIs that were identified correctly over all POIs in the dataset.

## Validation
Validation means using a a validation test set for tuning the parameters of a classifier. It is important to perform validation and tune the parameters in order to achieve better performance, but the results on a validation test set will always be biased since it's been using for tuning itself.

I have used cross validation technique that randomly splits the dataset into train and test subsets. 33% of the dataset has been used for testing.

## Final results
- Precision: 0.45833
- Recall: 0.34650

Full output:

```
$ python tester.py
Pipeline(steps=[('SKB', SelectKBest(k=9, score_func=<function f_classif at 0x10aaeeed8>)), ('PCA', PCA(copy=True, n_components=5, whiten=True)), ('GaussianNB', GaussianNB())])
	Accuracy: 0.85827	Precision: 0.45833	Recall: 0.34650	F1: 0.39465	F2: 0.36428
	Total predictions: 15000	True positives:  693	False positives:  819	False negatives: 1307	True negatives: 12181
```
