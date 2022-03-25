import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder



def preprocessing_german(df):
    # adding a derived sex attribute based on personal_status
    sexdict = {'A91' : 'male', 'A93' : 'male', 'A94' : 'male', 'A92' : 'female', 'A95' : 'female'}
    df = df.assign(personal_status = df['personal_status'].replace(to_replace = sexdict))
    df = df.rename(columns = {'personal_status' : 'sex'})

    df = df.replace({'sex':{"male": 0, "female": 1}})
    df = df.replace({'credit':{2: 0}})

    old = df['age'] >= 25
    df.loc[old, 'age'] = 'adult'
    young = df['age'] != 'adult'
    df.loc[young, 'age'] = 'youth'
    
    # one hot encoding
    categorical_columns = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'age', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
    enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    enc_df = pd.DataFrame(enc.fit_transform(df[categorical_columns]).toarray())
    df = df.join(enc_df)
    df = df.drop(columns=categorical_columns)
    return df

def preprocessing_credit_lending(df):

    df = df.replace({'SEX':{1: 0}})
    df = df.replace({'SEX':{2: 1}})
    df = df.rename(columns={'default.payment.next.month': 'def_payment_next_month', 'PAY_0': 'PAY_1'})
    fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[fil, 'EDUCATION'] = 4
    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

    fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)
    df.loc[fil, 'PAY_1'] = 0
    fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)
    df.loc[fil, 'PAY_2'] = 0
    fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)
    df.loc[fil, 'PAY_3'] = 0
    fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)
    df.loc[fil, 'PAY_4'] = 0
    fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)
    df.loc[fil, 'PAY_5'] = 0
    fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)
    df.loc[fil, 'PAY_6'] = 0

    df.loc[df.PAY_1 > 0, 'PAY_1'] = 1
    df.loc[df.PAY_2 > 0, 'PAY_2'] = 1
    df.loc[df.PAY_3 > 0, 'PAY_3'] = 1
    df.loc[df.PAY_4 > 0, 'PAY_4'] = 1
    df.loc[df.PAY_5 > 0, 'PAY_5'] = 1
    df.loc[df.PAY_6 > 0, 'PAY_6'] = 1

    df['SE_MA'] = 0
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 1)) , 'SE_MA'] = 1 #married man
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 2)) , 'SE_MA'] = 2 #single man
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 3)) , 'SE_MA'] = 3 #divorced man
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 1)) , 'SE_MA'] = 4 #married woman
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 2)) , 'SE_MA'] = 5 #single woman
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 3)) , 'SE_MA'] = 6 #divorced woman

    df['AgeBin'] = pd.cut(df['AGE'], 6, labels = [1,2,3,4,5,6])
    #because 1 2 3 ecc are "categories" so far and we need numbers
    df['AgeBin'] = pd.to_numeric(df['AgeBin'])
    df.loc[(df['AgeBin'] == 6) , 'AgeBin'] = 5

    return df

def preprocessing_algorithmic_hiring(df):
    # one hot encoding
    categorical_columns = ["workclass-previous-job", "education", "race","native-country"]
    enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    enc_df = pd.DataFrame(enc.fit_transform(df[categorical_columns]).toarray())
    df = df.join(enc_df)
    df = df.drop(columns=categorical_columns)
    return df


# Datasets
german_credit = {
    'tag': 'german_credit',
    'name': 'Credit Lending Data',
    'filename': 'german.csv',
    'sensitive_attribute': 'sex',
    'target': 'credit',
    'numerical_attributes': ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count'],
    'preprocessing_fct': preprocessing_german
}

credit_lending = {
    'tag': 'credit_lending',
    'name': 'Credit Lending Data',
    'filename': 'UCI_Credit_Card.csv',
    'sensitive_attribute': 'SEX',
    'target': 'def_payment_next_month',
    'numerical_attributes': ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count'],
    'preprocessing_fct': preprocessing_credit_lending
}

algorithmic_hiring = {
    'tag': 'algorithmic_hiring',
    'name': 'Algorithmic Hiring Data',
    'filename': 'algorithmic_hiring.csv',
    'sensitive_attribute': 'sex',
    'target': 'qualified',
    'numerical_attributes': [],
    'preprocessing_fct': preprocessing_algorithmic_hiring
}



def train(dataset_info):

    filename, target = dataset_info['filename'], dataset_info['target']

    preprocessing = dataset_info["preprocessing_fct"]

    df = preprocessing(pd.read_csv('data/' + filename))

    y = df[target]
    #y = y.replace(to_replace=2, value=0, inplace=False)
    X = df.drop([target], axis=1)

    """

    number_of_folds=5

    # Split data into training/holdout sets
    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
    kf.get_n_splits(X)

    # Keep track of the data for the folds
    folds = []

    # Iterate over folds, using k-1 folds for training
    # and the k-th fold for validation
    for train_index, test_index in kf.split(X):
        # Training data
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        
        # Holdout data
        X_test = X.iloc[test_index]
        y_test = y[test_index]
        
        
        # numerical=dataset_info['numerical_attributes']

        # #scale data
        # scaler = StandardScaler()

        # scaler.fit(X_train[numerical])
        
        # X_train[numerical] = scaler.transform(X_train[numerical])
        # X_test[numerical] = scaler.transform(X_test[numerical])


        A_train = X_train[dataset_info['sensitive_attribute']]
        A_test = X_test[dataset_info['sensitive_attribute']]

        fold = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'A_train': A_train,
            'A_test': A_test
        }

        folds.append(fold)

    fold = folds[0]
    X_train, X_test, y_train, y_test, A_train, A_test = fold['X_train'], fold['X_test'], fold['y_train'], fold['y_test'], fold['A_train'], fold['A_test']
    """


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    A_train = X_train[dataset_info['sensitive_attribute']]
    A_test = X_test[dataset_info['sensitive_attribute']]

    clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
    scores = clf.predict_proba(X_train)

    #filename_model = 'model.sav'
    #filename_scores = 'scores.sav'
    #pickle.dump(clf, open(filename_model, 'wb'))
    #pickle.dump(scores, open(filename_scores, 'wb'))

    print("The model has been trained.")

    print('Accuracy:', clf.score(X_test, y_test))
    predictions = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_true = y_test, y_pred = predictions))

    scores = np.array([score[1] for score in scores])
    y = y_train
    sensitive_attribute = A_train


    ix_A = sensitive_attribute == 0
    ix_B = sensitive_attribute == 1



    scores_json = {'scores_group1': scores[ix_A].tolist(), 'scores_group2': scores[ix_B].tolist()}
    json.dump(scores_json, open("output/" + dataset_info["tag"] + '/scores.json', 'w'))
    y_json = {'y_group1': y[ix_A].tolist(), 'y_group2': y[ix_B].tolist()}
    json.dump(y_json, open("output/" + dataset_info["tag"] + '/y.json', 'w'))

    print("Scores and y are saved in json format to output/" + dataset_info["tag"])

    return df, scores_json, y_json


if __name__ == "__main__":
    train(german_credit)
    #train(credit_lending)
    #train(algorithmic_hiring)