import csv
import random
import numpy as np
import pandas as pd

def encode_features(df_train, df_test):
    '''
    Takes columns whose values are strings (objects)
    and categorizes them into discrete numbers.
    This makes it feasible to use regression
    '''
    features = list(df_train.select_dtypes(include=['object']).columns)
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        unique_categories = list(df_combined[feature].unique())
        map_dict = {}
        for idx, category in enumerate(unique_categories):
            map_dict[category] = idx + 1
        df_train[feature] = df_train[feature].map(map_dict)
        df_test[feature] = df_test[feature].map(map_dict)

    return df_train, df_test


def cleanup(df):
    '''
    Cleans data
        1. Creates new features:
            - total bathrooms = full + half bathrooms
            - total porch area = closed + open porch area
        2. Drops unwanted features
        3. Fills missing values with the mode
        4. Performs feature scaling
    '''
    # Features to drop
    to_drop = ['MiscFeature', 'MiscVal', 'GarageArea', 'GarageYrBlt', 'Street', 'Alley',
              'LotShape', 'LandContour', 'LandSlope', 'RoofMatl', 'Exterior2nd', 'MasVnrType',
              'MasVnrArea', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
              'BsmtFinSF1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'Electrical',
              'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
              'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'FireplaceQu',
              'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolQC', 'MoSold']

    df['Bathrooms'] = df['FullBath'] + df['HalfBath']
    df['PorchSF'] = df['EnclosedPorch'] + df['OpenPorchSF']
    df = df.drop(to_drop, axis=1)

    # Columns to ignore when normalizing features
    to_ignore = ['SalePrice', 'Id']
    for column in df.columns:
        x = df[column].dropna().value_counts().index[0]
        df = df.fillna(x)
        if df[column].dtype != 'object' and column not in to_ignore:
            m = df[column].min()
            M = df[column].max()
            Range = M - m
            df[column] = (df[column] - m) / Range
    return df

def split(train_dataset):
    '''
    Shuffle data and split into 3 datasets
        1. Training - 80%
        2. Validation - 20%
        3. Testing - 20%
    '''
    # Shuffle data
    train_dataset = train_dataset.sample(frac=1)

    train, valid, test = np.split(train_dataset,
        [int(.6 * len(train_dataset)), int(.8 * len(train_dataset))])

    # Convert into numpy arrays
    x_train = train.drop(['SalePrice', 'Id'], axis=1).as_matrix().astype(np.float32)
    y_train = train['SalePrice'].as_matrix().astype(np.float32).reshape((np.shape(x_train)[0], 1))
    x_test = test.drop(['SalePrice', 'Id'], axis=1).as_matrix().astype(np.float32)
    y_test = test['SalePrice'].as_matrix().astype(np.float32).reshape((np.shape(x_test)[0], 1))
    x_valid = valid.drop(['SalePrice', 'Id'], axis=1).as_matrix().astype(np.float32)
    y_valid = valid['SalePrice'].as_matrix().astype(np.float32).reshape((np.shape(x_valid)[0], 1))

    return x_train, y_train, x_test, y_test, x_valid, y_valid
