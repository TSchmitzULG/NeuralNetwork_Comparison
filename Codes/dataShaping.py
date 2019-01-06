#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping loading in file at each batch, format [batchSize,numStep]"""
import scipy.io as sio
import numpy as np

# take input matrix train and input matrix test
def loadInputOutput(matrixTrain,matrixTest,num_step,maxSize):
    matrixIn = matrixTrain[:maxSize,0]
    matrixOut = matrixTrain[:maxSize,1]
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    trainInput = reshapedInput
    trainOutput = reshapedOutput
    
    matrixIn = matrixTest[:,0]
    matrixOut = matrixTest[:,1]
    testSize = len(matrixIn)
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    testInput = reshapedInput
    testOutput = reshapedOutput
    return trainInput,trainOutput,testInput,testOutput


# take input matrix and return shuffled train/test input/input
def splitShuffleData(matrix,num_step,trainTestRatio,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixOut = matrix[:maxSize,1]
    trainSize = int(len(matrixIn)*trainTestRatio)
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    inputShuffled = shuffleMatrix(reshapedInput)
    outputShuffled = shuffleMatrix(reshapedOutput)
    trainInput = inputShuffled[:trainSize]
    trainOutput = outputShuffled[:trainSize]
    testInput = inputShuffled[trainSize:]
    testOutput = outputShuffled[trainSize:]
    return trainInput,trainOutput,testInput,testOutput

def trainOnly(matrix,num_step,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixOut = matrix[:maxSize,1]
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    inputShuffled = shuffleMatrix(reshapedInput)
    outputShuffled = shuffleMatrix(reshapedOutput)
    return inputShuffled,outputShuffled

def shapeData(matrix,num_step,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixOut = matrix[:maxSize,1]
    reshapedInput  = []
    reshapedOutput =[]
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    return reshapedInput, reshapedOutput 

def shuffleMatrix(data):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    return [data[i] for i in shuffled_indices]

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

