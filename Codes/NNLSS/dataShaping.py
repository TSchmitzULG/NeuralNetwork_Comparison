#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping loading in file at each batch, format [batchSize]"""
import scipy.io as sio
import numpy as np

# take input matrix train and input matrix test
def loadInputOutput(matrixTrain,matrixTest,maxSize):
    
    trainInput = np.reshape(matrixTrain[:maxSize,0],(maxSize,1))
    trainOutput = np.reshape(matrixTrain[:maxSize,1],(maxSize,1))
    sizeTest = len(matrixTest)
    if maxSize < sizeTest: sizeTest=maxSize
    testInput = np.reshape(matrixTest[:sizeTest,0],(sizeTest,1))
    testOutput = np.reshape(matrixTest[:sizeTest,1],(sizeTest,1))
    return trainInput,trainOutput,testInput,testOutput

def loadValidation(matrix,valSize):
    maxSize = len(matrix)
    valInput = np.reshape(matrix[:valSize,0],(valSize,1))
    valOutput = np.reshape(matrix[:valSize,1],(valSize,1))
    
    return valInput,valOutput

def loadInputOutputSeq(matrixTrain,matrixTest,num_step,maxSize):
    
    trainInput = matrixTrain[:maxSize,0]
    my_indices = np.arange(len(trainInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    trainInput = np.take(trainInput,indices)
    
    trainOutput = np.reshape(matrixTrain[num_step-1:len(trainInput)+num_step-1,1],(len(trainInput),1))
    
    sizeTest = len(matrixTest)
    if maxSize < sizeTest: sizeTest=maxSize
    testInput = matrixTest[:sizeTest,0]
    my_indices = np.arange(len(testInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    testInput = np.take(testInput,indices)
    
    testOutput = np.reshape(matrixTest[num_step-1:num_step-1+len(testInput),1],(len(testInput),1))
    return trainInput,trainOutput,testInput,testOutput

def loadValidationSeq(matrix,num_step,valSize):
    maxSize = len(matrix)
    valInput = matrix[:valSize,0]
    my_indices = np.arange(len(valInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    valInput = np.take(valInput,indices)
    valOutput = np.reshape(matrix[num_step-1:num_step-1+len(valInput),1],(len(valInput),1))
    
    return valInput,valOutput


