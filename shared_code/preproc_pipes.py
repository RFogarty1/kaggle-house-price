
import numpy as np
import pandas as pd

import sklearn as sk
import sklearn.model_selection

import _ordinal_encode as _ordEncodeHelp
import _feat_eng as _featEngHelp

class RemoveOutliersById():

	def __init__(self, idVals):
		self.idVals = idVals

	def fit(self, inpX, inpY=None):
		return self

	def transform(self, inpX):
		outFrame = inpX.copy()
		for idx in self.idVals:
			outFrame.drop( outFrame.loc[outFrame["Id"]==idx].index, inplace=True )
		return outFrame 

#Useful for mutual information calculations
def getDiscreteFeatureNames():
	outFeats = ["Alley", "BedroomAbvGr", "BldgType", "BsmtCond", "BsmtExposure",
	            "BsmtFractUnfurnished", "BsmtFullBath","BsmtQual", "CentralAir", "Condition2",
	            "Electrical", "ExterCond", "ExterQual", "Exterior2nd", "Fence", "FenceQual", "FireplaceQu",
	             "Fireplaces", "FullBath", "Functional", "GarageCars", "GarageAreaTimesFinish", "GarageCond",
	             "GarageFinish", "GarageQual", "HalfBath", "Heating", "HeatingQC", "KitchenAbvGr",
	             "KitchenQual", "LandContour","LandSlope", "LotShape", "MSSubClass", "MasVnrType",
	             "MiscFeature", "MoSold", "NumbBath", "NumbBsmtBath",
	             "NumbStoreys_fromHouseStyle","OverallCond", "OverallQual",
	             "OverallQualTimesCond", "PavedDrive", "PoolQC", "PoolQualTimesCond",
	             "PorchLikeArea", "Spaciousness", "Street", "TotRmsAbvGrd", "TotalSFLiv",
	             "TotalSFLivOverLotArea", "Utilities", "YearBuilt", "YearRemodAdd",
	             "YrSold", "YearSold_Fract"
			   ]
	return outFeats


class FactorizeRemainingCateGroups():

	def __init__(self):
		pass

	def fit(self, inpX, inpY=None):
		self.useCols = inpX.select_dtypes(["object"]).columns
		self.mapDict = dict()
		for key in self.useCols:
			unused, mapVals = pd.factorize(inpX[key])
			currDict = {key:idx for idx,key in enumerate(mapVals)}
			self.mapDict[key] = currDict		

		return self

	def transform(self, inpX):
		outFrame = inpX.copy()
		
		nanVal = -1
		#Note: For unrecognised values we put them in a group 1 greater than the highest idx
		def _mapFunct(inpVal):
			if pd.isna(inpVal):
				return nanVal
			unknownVal = max(currDict.values()) + 1
			return currDict.get(inpVal, unknownVal)
		
		for key in self.useCols:
			currDict = self.mapDict[key]
			outFrame[key] = outFrame[key].map(_mapFunct)
			
		return outFrame 




class OrdinalEncoder():
    
    def __init__(self, encodeKeys):
        self.encodeKeys = encodeKeys

    def fit(self, inpX, inpY=None):
        return self
    
    def transform(self, inpX):
        outFrame = inpX.copy()
        for key in self.encodeKeys:
            _ordEncodeHelp.applyOrdinalEncodingForFrame(outFrame, key)
        return outFrame
    
def getStandardOrdinalEncodeKeys():
    outList = ["Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual",
               "Electrical", "ExterCond", "FireplaceQual", "Functional", "GarageCond", 
               "GarageFinish", "GarageQual", "KitchenQual", "LandSlope", "LotShape", "MiscFeature", "PoolQC"
              ]
    return outList



class EngFeatureAdder():

	def __init__(self, addFeatKeys):
		self.addFeatKeys = addFeatKeys

	def fit(self, inpX, inpY=None):
		return self

	def transform(self, inpX):
		outFrame = inpX.copy()
		for key in self.addFeatKeys:
			_featEngHelp.addFeatToFrame(outFrame,key)
		return outFrame



#
class TargEncoderMEst_Kfold():
	
	def __init__(self, nFolds, features, mVal=1):
		self.nFolds = nFolds
		self.mVal = mVal
		self.features = features
	
	def fit(self, inpX, y=None):
		self.inpFitArray = inpX.copy().to_numpy()
		self.trainFitArray = inpX.copy()
		self.globalMean = inpX["SalePrice"].mean()

		#Initialise columns for trainFitArray
		for feat in self.features:
			newName = feat + "_m{}".format(self.mVal)
			self.trainFitArray[newName] = self.trainFitArray[feat].copy()

		#For each k-fold we calculate mean values on the train; and use them to encode the test
		splitter = sk.model_selection.KFold(n_splits=self.nFolds)
		for trainIndices, testIndices in splitter.split(inpX):
			for feature in self.features:
				self._encodeSingleFeatureOnCurrentSplit(inpX, feature, trainIndices, testIndices)
	
		#Get a mean for each
		self.globalMeanDict = dict()
		for currFeat in self.features:
			self.globalMeanDict[currFeat] = inpX[currFeat].mean()
	
		#We need to figure out the encoding for each category; we do this by averaging the k-fold values of each
		self.encodeDict = dict()
		for currFeat in self.features:
			merged = inpX[[currFeat]].join( self.trainFitArray[[currFeat]], rsuffix="_enc" )
			self.encodeDict[currFeat] = merged.groupby(currFeat).mean()[currFeat+"_enc"]
	
		return self
	
	def _encodeSingleFeatureOnCurrentSplit(self, inpX, feature, trainIndices, valIndices):
		#1) Figure our the encoding values for this feature
		useX = inpX[[feature,"SalePrice"]]
		cateMeans = useX.iloc[trainIndices].groupby(feature).mean()["SalePrice"]
		cateN = useX.iloc[trainIndices][feature].value_counts()
		cateWeights = cateN / (cateN+self.mVal)
		globalMean = useX["SalePrice"].mean()
		encodeVals = (cateWeights*cateMeans) + (1-cateWeights)*globalMean

		#2) 
		newVals = useX.iloc[valIndices][feature].map( lambda x: encodeVals.get(x,globalMean))
		outName = feature + "_m{}".format(self.mVal)
		self.trainFitArray[outName].iloc[valIndices] = newVals

		
	def transform(self,inpX):
		#We want to pass the training data back with different encoding on each fold
		numpyRep = inpX.to_numpy()
		if numpyRep.shape == self.inpFitArray.shape:
			if np.allclose(numpyRep, self.inpFitArray):
				return self.trainFitArray.copy()
			
		#We want to encode any non-training data (using a single, average encoding)
		outX = inpX.copy()
		for feat in self.features:
			outName = feat + "_m{}".format(self.mVal)
			globalMean = self.globalMean
			currDict = self.encodeDict[feat]
			outX[outName] = inpX[feat].map( lambda x: currDict.get(x,self.globalMean) )
	
		return outX



