import torch
import torch.nn as nn

"""testX = torch.ones((5,5))
testX[4,0] = 0
testX[3,0] = 0
testY = torch.ones(5)
testYh = torch.ones(5)
testYh[4] = 0
testYh[2] = 0
testK = 0
print("Truth : " + str(testY))
print("Predi : " + str(testYh))
print("Prote : " + str(testX[:,testK]))"""

def AbsEqOppDiff(x,y,y_hat,k):
    a = x[:,k]
    indexY1 = y.int() == 1
    indexA1Y1 = (a * indexY1).bool()
    Pa1 = torch.sum(y_hat[indexA1Y1].int() == 1) / torch.sum(indexA1Y1)
    indexA0Y1 = ((1-a) * indexY1).bool()
    Pa0 = torch.sum(y_hat[indexA0Y1].int() == 1) / torch.sum(indexA0Y1)
    return torch.abs(Pa0 - Pa1)

def AbsAvgOddsDiff(x,y,y_hat,k):
    return 0.5 * (AbsEqOppDiff(x,y,y_hat,k) + AbsEqOppDiff(x,1-y,y_hat,k))

def DisparateImpact(x,y,y_hat,k):
    #A revoir, ça depend pas du modele pour le moment...
    a = x[:,k].bool()
    indexYh = y_hat == 1
    num = torch.sum(~a * indexYh) / torch.sum(~a)
    denom = torch.sum(a * indexYh) / torch.sum(a)
    return 1 - num/denom

"""
print("###TESTING")
print(AbsEqOppDiff(testX,testY,testYh,testK))
print(AbsAvgOddsDiff(testX,testY,testYh,testK))
print(DisparateImpact(testX,testY,testYh,testK))
raise E"""