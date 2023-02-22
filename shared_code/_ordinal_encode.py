

""" Code to encode ordinal values """

import pandas as pd
import _registration_funct as _regFunctHelp


_ENCODE_DICT = dict()
regEncodeFunct = _regFunctHelp.RegisterKeyValDecorator(_ENCODE_DICT)

def getRegisteredKeys():
	return _ENCODE_DICT.keys()

def applyOrdinalEncodingForFrame(inpFrame, inpKey):
	_ENCODE_DICT[inpKey](inpFrame)

#This could be EASILY extended by addition of a "defKey" for handling anything
#in test data that doesnt appear in train data
def _getStandardMappedVals(inpFrame, featName, mapDict, nanKey="None"):
	mapped = inpFrame[featName].map(lambda x: nanKey if pd.isna(x) else x)
	mapped = mapped.map( lambda x:mapDict[x] )
	return mapped

#Specific encodings below
@regEncodeFunct("Alley")
def _unused(inpFrame):
	mapDict = {"None":0, "Grvl":1, "Pave":2}
	inpFrame["Alley"] = _getStandardMappedVals(inpFrame, "Alley", mapDict)

@regEncodeFunct("BsmtCond")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["BsmtCond"] = _getStandardMappedVals(inpFrame, "BsmtCond", mapDict)

@regEncodeFunct("BsmtExposure")
def _unused(inpFrame):
	mapDict = {"None":0, "No":1, "Mn":2, "Av":3, "Gd":4}
	inpFrame["BsmtExposure"] = _getStandardMappedVals(inpFrame, "BsmtExposure", mapDict)

@regEncodeFunct("BsmtFinType1")
def _unused(inpFrame):
	mapDict = {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6}
	inpFrame["BsmtFinType1"] = _getStandardMappedVals(inpFrame, "BsmtFinType1", mapDict)

@regEncodeFunct("BsmtFinType2")
def _unused(inpFrame):
	mapDict = {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6}
	inpFrame["BsmtFinType2"] = _getStandardMappedVals(inpFrame, "BsmtFinType2", mapDict)

@regEncodeFunct("BsmtQual")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["BsmtQual"] = _getStandardMappedVals(inpFrame, "BsmtQual", mapDict)

@regEncodeFunct("Electrical")
def _unused(inpFrame):
	mapDict = {"None":0, "FuseP":1, "Mix":2, "FuseF":3, "FuseA":4, "SBrkr":5}
	inpFrame["Electrical"] = _getStandardMappedVals(inpFrame, "Electrical", mapDict)

@regEncodeFunct("ExterCond")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["ExterCond"] = _getStandardMappedVals(inpFrame, "ExterCond", mapDict)

@regEncodeFunct("ExterQual")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["ExterQual"] = _getStandardMappedVals(inpFrame, "ExterQual", mapDict)

@regEncodeFunct("FireplaceQual")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["FireplaceQu"] = _getStandardMappedVals(inpFrame, "FireplaceQu", mapDict)

@regEncodeFunct("Functional")
def _unused(inpFrame):
	mapDict = {"Sal":0, "Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min2":5, "Min1":6, "None":7,"Typ":7}
	inpFrame["Functional"] = _getStandardMappedVals(inpFrame, "Functional", mapDict)

@regEncodeFunct("GarageCond")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["GarageCond"] = _getStandardMappedVals(inpFrame, "GarageCond", mapDict)

@regEncodeFunct("GarageFinish")
def _unused(inpFrame):
	mapDict = {"None":0, "Unf":1, "RFn":2, "Fin":3}
	inpFrame["GarageFinish"] = _getStandardMappedVals(inpFrame, "GarageFinish", mapDict)

@regEncodeFunct("GarageQual")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["GarageQual"] = _getStandardMappedVals(inpFrame, "GarageQual", mapDict)

@regEncodeFunct("HeatingQC")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "Gd":3, "TA":4, "Ex":5}
	inpFrame["HeatingQC"] = _getStandardMappedVals(inpFrame, "HeatingQC", mapDict)

@regEncodeFunct("KitchenQual")
def _unused(inpFrame):
	mapDict = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
	inpFrame["KitchenQual"] = _getStandardMappedVals(inpFrame, "KitchenQual", mapDict)

@regEncodeFunct("LandSlope")
def _unused(inpFrame):
	mapDict = {"Gtl":0, "Mod":1, "Sev":2}
	inpFrame["LandSlope"] = _getStandardMappedVals(inpFrame, "LandSlope", mapDict)

@regEncodeFunct("LotShape")
def _unused(inpFrame):
	mapDict = {"None":0, "IR3":1, "IR2":2, "IR1":3, "Reg":4}
	inpFrame["LotShape"] = _getStandardMappedVals(inpFrame, "LotShape", mapDict)

@regEncodeFunct("MiscFeature")
def _unused(inpFrame):
	mapDict = {"None":0, "Othr":1, "Shed":2, "Elev":3, "Gar2":4, "TenC":5}
	inpFrame["MiscFeature"] = _getStandardMappedVals(inpFrame, "MiscFeature", mapDict)

@regEncodeFunct("PoolQC")
def _unused(inpFrame):
	mapDict = {"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
	inpFrame["PoolQC"] = _getStandardMappedVals(inpFrame, "PoolQC", mapDict)




