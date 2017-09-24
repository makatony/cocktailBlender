#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:53:47 2017

@author: soojunghong

# Cocktail Blender using word2vec 
"""

import json    
import numpy as np
import pandas

#-------------------------------------
# create dataframe using two column 
#-------------------------------------
def createCocktailDataFrame(): 
    columns = ['cocktail', 'instruction']
    index = np.arange(0)
    df = pandas.DataFrame(columns=columns, index=index)
    return df

#-----------------------------------------
# Extract cocktail name and instruction
#-----------------------------------------
import pandas as pd
def addNameAndInstr(inputDiction, df): 
        t = pd.DataFrame(inputDiction.items())
        t = t.T
        print(t[37][1]) #37 is 'strDrink' name 
        print(t[8][1]) #column value
       # df.append({'cocktail':t[37][1], 'instruction':t[8][1]}, ignore_index=True)
        df.loc[len(cocktailDataFrame)] = [t[37][1], t[8][1]]
        print(df)
        return

#-------------------------------------
# read json file (cocktail db files) 
#-------------------------------------
def readCocktailFile(fileName): 
 #fileName = '/Users/soojunghong/cocktailBlender/cocktaildb/11001.json'
 config = json.loads(open(fileName).read())
 return config  

#----------------------------
# go through all json files
#----------------------------
import os.path
def fillDataFrame(newDF): 
    for num in range(11000, 17230): 
        numStr = str(num)
        fileName = '/Users/soojunghong/cocktailBlender/cocktaildb/' + numStr + '.json'
        if(os.path.exists(fileName)): 
            currentFile = readCocktailFile(fileName)
            addNameAndInstr(currentFile, newDF)
            print(newDF)
        else:
            print(numStr + 'file does not exist')
    return newDF

#----------------------------------------------------------------------------------
# construct data frame using 'strDrink' as a key and 'strInstruction' as a value  
#----------------------------------------------------------------------------------
cocktailDataFrame = createCocktailDataFrame()
cocktailDataFrame
fillDataFrame(cocktailDataFrame)

#---------------------------------
# from dataframe to list of rows 
#---------------------------------
def dfToList(df, wholeList): 
    for index, row in df.iterrows():
        #print row["cocktail"], row["instruction"]
        recipe = [row["cocktail"], row["instruction"]]
        print(recipe)    
        wholeList.append(recipe)
    return wholeList
        
     
wholeList = []
cocktail_instruct = dfToList(cocktailDataFrame, wholeList)

    
#-------------------------------------------------
# ToDo : Apply the cocktail_instruct to word2vec 
#-------------------------------------------------
# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sentences = [['first', 'sentence'], ['second', 'sentence']]
print(type(sentences))
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)  
model  
some_sentences = [['cocktail1', 'receipe1'], ['cocktail2', 'receipe2']]
other_sentences = [['cocktail3', 'receipe3'], ['cocktail4', 'receipe4']]

model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)  # can be a non-repeatable, 1-pass generator

model['cocktail2']  # raw NumPy vector of a word
model.similarity('cocktail1', 'cocktail2')

