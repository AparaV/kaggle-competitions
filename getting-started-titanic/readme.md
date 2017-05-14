# Titanic: Machine Learning from Disaster

## Data

Check this [website](https://www.kaggle.com/c/titanic/data) for raw data

### Cleanup
- To prevent overfitting, the following features were discretized into bins
   - Age into `['Child', 'Teen', 'Adult', 'Senior']`
   - Fare into `['1_quartile', '2_quartile', '3_quartile', '4_quartile']`
- Missing ages were filled with mean values
- Missing fares (fares with value 0) were filled with mean values of corresponding travel class
- Missing ports of embarkment were filled with the mode of port of embarkment
- Names were split into last name and prefix
- Only the first letter of cabins were used. Additional cabins (if present) and cabin numbers were dropped
- Ticket number and full names were dropped
- Features with string values (`['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix', 'Embarked']`) were encoded

## Models used for training
### Support Vector Machine

Used the `svm.SVC()` from `sklearn` package. The entire training dataset was used to train the classifier. Had an accuracy of **65.778 %**

### Random Forest Classifier

Used the `RandomForestClassifier` from `sklearn` package. Different parameters were used for training the Random Forest classifier. 80% of the training data was used to train and the other 20% was used to test and report error.

```python
parameters = {'n_estimators': [9, 50, 250],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
```

The classifier was also tested with KFold analysis by dividing the dataset into 10 separate bins.

Had an accuracy of **75.558 %**

## Remarks
There is scope for improvement here. With more data visualization, we could better engineer our features or create new features for training. We could also use other classifiers or play with the parameters of SVMs and RandomForestClassifiers to get more accurate results.

Upon closer inspection, it can be seen that many of the top scorers in the leaderboard, heavily overfit their parameters to produce *unreal* results. Such parameters are not of much use outside the dataset (and open a whole new topic on ethics).

## Citations
The data cleanup methodology and the usage of Random Forest Classifier was adapted from and inspired by [Jeff Delaney's tutorial](https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish)
