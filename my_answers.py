import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re


### DONE: fill out the function below that transforms the input series 
### and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(0, len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])   

    # reshape each (i.e. change y from (131,) to (131, 1))
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    ### DONE: create required RNN model

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # DONE: build an RNN to perform regression on our time series input/output data

    # layer 1 of Sequential model uses LSTM module with:
    # - 5 hidden units
    # - input_shape = (window_size,1))
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))

    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, 
                                         rho=0.9, 
                                         epsilon=1e-08,
                                         decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', 
                  optimizer=optimizer)

    return model


### DONE: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique = ''.join(list(set(text)))
    print('unique before cleanse: ', unique)

    # remove as many non-english characters and character sequences as you can 

    # with the exception of necessary punctuation marks

    # testing test
    # text = 'èéà&%$*@â\'[](){}⟨⟩:,-!.‹›«»?‘’“”\'\'";\\/ -- \"test;\" test. test? \'test\'"'

    # A - DONE replace punctuation i.e.: ‘ --> '
    # B - DONE replace punctuation i.e.: “ --> "
    # C - DONE remove non-english chars i.e.: èéà&%$*@â\
    # D - DONE remove non-english chars escaped i.e.: \/
    # E - DONE remove undesired punctuation i.e.: [](){}⟨⟩--‹›«»
    # F - DONE retain desired punctuation escaped before/after word i.e.: \'\"
    # retain desired punctuation after word before space i.e.: ,!.?;
    # retain desired punctuation between non-space characters i.e.: -

    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "\'").replace("’", "\'")
    text = re.sub(r'[^a-zA-Z!.?\-,:;\"\']', ' ', text)
    text = text.replace("--", "")
    # remove fullstops just infront of a word
    text = re.sub(r'!.\b(?!\w)', ' ', text)
    # remove other fullstops not immediately before/after word
    text = re.sub(r'!.(?!\w)', ' ', text)
        
    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    unique = ''.join(list(set(text)))
    print('unique after cleanse: ', unique)

    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    

    
    return inputs,outputs
