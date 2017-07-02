import pandas as pd



if __name__ == "__main__":

    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')

    print data_train.columns
