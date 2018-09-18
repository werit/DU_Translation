import numpy as np

def fnLstOrd(string):
    return [ord(letter) for letter in string]

def Sigmoid(vector):
    '''
    @vector is an array-like structure
    '''
    return 1/(1+np.exp(-vector))

pol = np.array([['Ahoj'],['ako'],['sa']])
print([lt for lt in pol])
print(np.array([0,0]))

# features = np.array([[2,3],[5,6],[7,10]])
# wg = np.zeros(features.shape[1])
# targ = np.array([0,1,1])
# pred = Sigmoid(np.dot(features,wg))
# err = targ-pred
# print(f'Features are: {features}')
# print(f'Predictions are: {pred}')
# print(f'Error is: {err}')
# print(np.dot(features.T,err))
