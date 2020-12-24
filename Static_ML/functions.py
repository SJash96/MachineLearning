from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import sys
import numpy as np
from Dict import filenames, attacks, underSample, overSample

def memoryOptimization(data):
    
    #Downcast float datatypes to the smallest possible datatype without losing information
    floats = data.select_dtypes(include=['float64']).columns.tolist()
    data[floats] = data[floats].apply(pd.to_numeric, downcast='float')

    #Downcast int datatypes to the smallest possible datatype without losing information
    ints = data.select_dtypes(include=['int64']).columns.tolist()
    data[ints] = data[ints].apply(pd.to_numeric, downcast='integer')

    #Downcast object datatypes to the smallest possible datatype without losing information
    objects = data.select_dtypes(include=['object']).columns.tolist()
    for o in objects:
        num_unique_values = len(data[o].unique())
        num_total_values = len(data[o])
        if float(num_unique_values) / num_total_values < 0.5:
            data[o] = data[o].astype('category')

    dtypes = data.dtypes

    colnames = dtypes.index
    types = [t.name for t in dtypes.values]
    
    return dict(zip(colnames, types))


def readCSV(): 
    pd.options.display.max_rows = 1000
    pd.options.display.max_columns = 79
    
    #Read sample 1000 rows to get proper dtypes on each column
    data_NOToptimized = pd.read_csv('DataFiles/Th2.csv', low_memory=False, nrows=1000)

    #Apply memory optimization to all the data
    #In total about 85% memory was reduced just by changing datatypes!!!
    data_Optimized = pd.read_csv('DataFiles/combinedData.csv', low_memory=False, dtype=memoryOptimization(data_NOToptimized))

    #These columns have NULL/NAN values and also are of type object
    #Dropping the following columns in the dataframe
    data_Optimized = data_Optimized.drop(columns=['Flow Bytes/s', ' Flow Packets/s'])

    #Combine all csv files into one for a better ML experience
    #combined_CSV = pd.concat([pd.read_csv('DataFiles/'+files, low_memory=False, dtype=memoryOptimization(data_NOToptimized)) for files in filenames])
    #combined_CSV.to_csv('DataFiles/combinedData.csv', index=False, encoding='utf-8-sig')

    return data_Optimized

def printMetatdata(data):
    
    print(data.info(memory_usage='deep'))
    print("\nData Shape:\n", data.shape)
    data = data.sort_index(axis=1)
    print("\nSample Data:\n", data.head())
    print("\nTypes:\n", data[' Label'].value_counts())

#No need to resample the data in the file as it already gives good accuracy readings  
# def reSample(X, Y):
    
#     downSample = RandomUnderSampler(sampling_strategy=underSample)
#     X_downSamp, Y_downSamp = downSample.fit_resample(X, Y)

#     upSample = RandomOverSampler(sampling_strategy=overSample)
#     X_upSamp, Y_upSamp = upSample.fit_resample(X_downSamp, Y_downSamp)

#     return X_upSamp, Y_upSamp

def splitData(data, le):
    
    X = data[data.columns.difference([' Label'])]
    Y_Attk_Str = data[' Label']

    le.fit(Y_Attk_Str)
    Y = le.transform(Y_Attk_Str)

    #Test size: 70% is for training and 30% is for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    return X_train, X_test, Y_train, Y_test

def trainOnDecTree(X_train, Y_train):

    #Running with default parameters
    decTree = DecisionTreeClassifier()
    decTree.fit(X_train, Y_train)

    return decTree

def scaleData(X_train, X_test, scaler):

    #Scaling the data for use in neural network
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def trainOnNeurNet(X_train, Y_train):

    #Running with default parameters
    mlpNeurlNet = MLPClassifier()
    mlpNeurlNet.fit(X_train, Y_train)

    return mlpNeurlNet

def predictions(X_test, classifier, le, X_test_Orig=None):

    Y_pred = classifier.predict(X_test)
    print("\nPrediction Outcome:")

    labels = list(le.inverse_transform(Y_pred))
    labelK = list(Counter(labels).keys())
    labelV = list(Counter(labels).values())
    labelDic = {labelK[i]: labelV[i] for i in range(len(labelK))}

    Y_leK = list(Counter(Y_pred).keys())
    Y_leV = list(Counter(Y_pred).values())
    labelEncDic = {Y_leK[j]: Y_leV[j]  for j in range(len(Y_leK))}

    print("\nLabel Names: ")
    [print(str(k) + "\t" + str(v)) for k, v in labelDic.items()]

    if X_test_Orig is not None:
        sampleDF = pd.DataFrame(X_test_Orig)
        sampleDF[' Label'] = labels
        print("\nSample Predicitons: ")
        print(sampleDF.head())
    else:
        sampleDF = pd.DataFrame(X_test)
        sampleDF[' Label'] = labels
        print("\nSample Predicitons: ")
        print(sampleDF.head())

    #print("\nEncoded Names: ")
    #[print(k_le, v_le) for k_le, v_le in labelEncDic.items()]

    return Y_pred, list(labelEncDic.keys()), list(labelDic.keys())

def accuracyCalc(Y_test, Y_pred, lbl_keys, lbl_names=None):

    print("\nConfusion Matrix:\n ", confusion_matrix(Y_test, Y_pred, labels=lbl_keys))
    print("\nAccuracy: ", accuracy_score(Y_test, Y_pred) * 100)
    print("\nReport:\n ", classification_report(Y_test, Y_pred))