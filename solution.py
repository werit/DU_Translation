import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.linalg import inv

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
    return 1/(1+np.exp(-vector))

# tu spravim logisticku regresiu
def TaskTwo(df_sk,df_hu):
    '''
    Mix the data
    Calculate Logistic regresion
    '''
    print(Sigmoid(np.dot([-1,2,5],[[2],[3],[8]])))
    pass

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
        TaskTwo(df_sk=dframe_sk,df_hu=dframe_hu)
    else:
        print('Dataframes do not match!')



# spustenie main metody
if __name__ == '__main__':
    main()