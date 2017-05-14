import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import svm


# Create bins for ages - replace NaN with average value
def simplify_ages(df):
    df.Age = df.Age.fillna(df["Age"].mean())
    bins = (0, 10, 20, 65, 80)
    group_names = ['Child', 'Teen', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

# Get rid of cabin number - also ignores multiple cabins
# Keep only the first letter
# Fill NaN with 'N'
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

# Create bins for fares
# Treat 0 fare as NaN and replace with mean of corresponding class
def simplify_fares(df):
    df.Fare = df.Fare.replace(0, np.nan)
    df.Fare = df.Fare.fillna(df.groupby("Pclass").Fare.transform("mean"))
    bins = (0, 8, 15, 31, 1000)
    group_names = ['1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

# Fill NaNs
def format_embarked(df):
    df.Embarked = df.Embarked.fillna(df.Embarked.mode())
    return df

# Split name into prefix ("Mr.", "Mrs.", "Dr.") and last name
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

# Drop ticket, name and port of embarkment features
def drop_features(df):
    return df.drop(['Ticket', 'Name'], axis=1)

# Clean data with helper functions
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_embarked(df)
    df = format_name(df)
    df = drop_features(df)
    return df

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix', 'Embarked']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))


if __name__ == "__main__":

    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)
    data_train, data_test = encode_features(data_train, data_test)

    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    # Split training data into 80% training and 20% testing
    num_test = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

    # Choose the type of classifier.
    clf = RandomForestClassifier()

    # Choose some parameter combinations to try
    parameters = {'n_estimators': [9, 50, 250],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1,5,8]
                 }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    run_kfold(clf)

    ids = data_test['PassengerId']
    predictions = clf.predict(data_test.drop('PassengerId', axis=1))

    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
    output.to_csv('titanic-predictions.csv', index = False)
