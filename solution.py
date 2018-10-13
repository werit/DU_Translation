import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.linalg import inv
import os


# metoda na citanie suboru sk.dic
def ReadFileToDataFrame(file_name,do_filtering=True):
    '''
    file_name: full path with name of file to open and read
    expected encoding is UFT-16-LE
    '''

    # obchadzam to, ze nemam oba subory v rovnakom kodovani....
    if file_name == 'sk.dic':
        enc = 'UTF-16-LE'
    else:
        enc = 'UTF-16'
    #file_name = 'sk.dic'
    # get working directory
    #cwd = os.getcwd()
    #print(cwd)

    with open(file_name,'r', encoding=enc) as f:
        cont = f.readlines()
        if do_filtering:
            # nacitany slovnik osekam len na slova, ktore maju aspon dva znaky a zacinaju na 's'
            df = DataFrame([word.strip() for word in cont if str(word).startswith('s') and len(word) > 1],columns=['word'])

        else:
            # vratim vsetky slova, nerobim filtrovanie
            df = DataFrame([word.strip() for word in cont],columns=['word'])

        df['length'] = df['word'].apply(len)
        return df


# metoda kontroluje datadramy, ci maju pozadovane valsntosti
def DoCheck(df1,df2):
    '''
    @df1 First DataFrame to compare to df2
    @df2 Second DataFrame used as comparison to df1
    @return Functions returns True only if all checks pass successfully.

    Checks performed:
    1. Length of both dataframes are equal....every word has its translation
    '''
    # overall check result
    pass_value = True
    if df1['word'].count() == df2['word'].count():
        print('OK, dlzka je rovnaka.')
        pass_value = pass_value and True
    else:
        print('Error, dlzka je rozdielna!')
        pass_value = False
    # return result of checks
    return pass_value

# tu spravim linearnu regresiu
def TaskOne(df_sk,df_hu):
    '''
    '''
    # zapojim vedla seba maticu dlzky slova a maticu samych jednotiek
    # hstack stackuje horizontalne
    # np.ones tvori maticu len s jednotkami a matica je taka velka, kolko je riadkov v df_sk
    matrix =np.hstack((np.matrix(df_sk['length'].values).T,np.ones((df_sk['length'].count(),1))))

    # do the estimation based on LSM
    estim = ( inv(matrix.T @ matrix) @ matrix.T ) @ (np.matrix(df_hu['length'].values).T)
    print(f'Vyhodnotenie odhadu linearnej regresie (uloha 1) je:')
    # tu prehadzujem Beta_0 a Beta_1, pretoze som pridaval jednotky na koniec a nie na zaciatok
    print(f'Beta_0 je {estim[1]}')
    print(f'Beta_1 je {estim[0]}')

def Sigmoid(vector):
    '''
    @vector is an array-like structure
    '''
    return 1/(1+np.exp(-vector))

def MinMaxTrans(number, min, max):
    '''
    Function performin min max normalization
    '''
    return (number - min)/(max - min)

def fnLstOrd(string):
    min = 0
    max = 2**16
    # 30 mam ako za najdlhsie slovo...nemal by som to pouzivat takto
    return np.append(np.array([MinMaxTrans(ord(letter),min,max) for letter in string]),np.zeros(30-len(string)),0)

def PrepareFeatures(df_sk,df_hu):
    '''
    Metoda pripravy slova zo slovnika, ako podklad pre log regresiu
    '''

    feat = df_sk[0]
    for i in range(df_sk.count()-1):
        feat = np.row_stack((feat,df_sk[i+1]))
    for i in range(df_hu.count()):
        feat = np.row_stack((feat,df_hu[i]))

    # dokoncenie features
    np.random.seed(42)
    perm = np.random.permutation(len(feat))

    intercept = np.ones((feat.shape[0],1))
    feat = np.concatenate((intercept,feat),axis=1)
    # tvorim permutaciu nad zdrojovymi datami
    feat = feat[perm]
    # 20% delenie train a test
    index_split = int((len(feat)/5)*4)
    feat_train = feat[:index_split]
    feat_test = feat[index_split:]

    # sk je 0
    zr = np.zeros(df_sk.count())
    # hu je 1
    on = np.ones(df_hu.count())

    target = np.append(zr,on)[perm]

    #index_split = int((len(target)/5)*4)
    target_train = target[:index_split]
    target_test = target[index_split:]
    return feat_train,feat_test,target_train,target_test


def LogReg(features,targets,weights,learning_rate,repeats):
    '''
    '''
    # print(f'Features are: {features.T}')
    for i in range(repeats):
        scores = np.dot(features, weights)
        # print(f'Scores are: {scores}')
        predictions = Sigmoid(scores)
        # print(f'Predic are: {predictions}')
        # print(f'Targ are: {targets}')
        # Update weights with gradient
        output_error_signal = targets - predictions
        # print(f'OUT err: {np.sum(output_error_signal)}')
        # print(f'Feeat trans shape is: {features.T.shape}')
        # print(f'output_error_signal shape is: {output_error_signal.shape}')
        # x = 0
        gradient = np.dot(features.T, output_error_signal)
        # print(f'Feat End: {features.T}')
        # print(f'Err End: {output_error_signal[x:]}')
        # print(f'Grad are: {gradient}')
        # print(f'Grad shape is: {gradient.shape}')
        weights += learning_rate * gradient

    return weights

def Accur(pred,target):
    '''
    Accuracy calculation
    @pred is a vector of predictions
    @target is a vector of target class
    '''
    miss_class = np.sum(np.abs(pred - target))
    acc = miss_class/len(target)
    return acc


# tu spravim logisticku regresiu
def TaskTwo(df_sk,df_hu,learning_rate):
    '''
    Mix the data
    Calculate Logistic regresion
    '''

    df_sk['ch_rep'] = df_sk['word'].apply(fnLstOrd)
    df_hu['ch_rep'] = df_hu['word'].apply(fnLstOrd)
    # PrepareFeatures(df_sk['ch_rep'],df_hu['ch_rep'])
    features_train,features_test,targets_train,targets_test = PrepareFeatures(df_sk['ch_rep'],df_hu['ch_rep'])

    # print(f'Features shape is: {features.shape}')
    # set weights to zero
    weights = np.zeros(features_train.shape[1])

    model_weights = LogReg(features=features_train,
                            targets=targets_train,
                            weights=weights,
                            learning_rate=0.01,
                            repeats=4000)
    print(f'Final weights are: {model_weights}')
    ###########################
    ## test nauceneho modelu ##
    ###########################

    model_acc = Accur(Sigmoid(np.dot(features_train, model_weights))>=0.5,targets_train)
    print(f'Trainning acc is: {model_acc}')
    scores = np.dot(features_test, model_weights)
    pred_test = Sigmoid(scores)>=0.5
    # print(f'Predictions {pred_test}')
    # print(f'Target {targets_test}')
    test_acc = Accur(pred_test,targets_test)
    print(f'Test acc is: {test_acc}')


# main metoda
def main():
    '''
    Main method
    1. read dictionary
    2. output sk dic for translation
    '''
    # construct
    dframe_sk = ReadFileToDataFrame('sk.dic')
    dframe_hu = ReadFileToDataFrame('HuWords.txt',do_filtering=False)

    # save DataFrame
    # already did
    # dframe.to_csv('SkWords.csv',encoding='UTF-16-LE',columns=['word'],index=False, header=False)

    # do check
    # if check is ok then continue
    if DoCheck(dframe_hu,dframe_sk):
        TaskOne(df_sk=dframe_sk,df_hu=dframe_hu)
        TaskTwo(df_sk=dframe_sk,df_hu=dframe_hu,learning_rate=0.5)
    else:
        print('Dataframes do not match!')



# spustenie main metody
if __name__ == '__main__':
    main()
