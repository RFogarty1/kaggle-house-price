
import numpy as np

import pandas as pd
import sklearn as sk
import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing

#
class StandardScaler():
	""" Basically a wrapper for the sklearn StandardScaler; this returns a data frame rather than a numpy array """
	
	def __init__(self, ignoreCols=None):
		self.ignoreCols = ignoreCols
	
	def fit(self, inpX, inpY=None):
		self.useDict = dict()
		for col in inpX.columns:
			inclVal = True
			if self.ignoreCols is not None:
				if str(col) in self.ignoreCols:
					inclVal = False
			if inclVal:
				self.useDict[col] = sk.preprocessing.StandardScaler()
				self.useDict[col].fit(inpX[col].to_numpy().reshape(-1, 1)  )

		return self
	
	def transform(self, inpX, inpY=None):
		outX = inpX.copy()
		for col in inpX.columns:
			inclVal = True
			if self.ignoreCols is not None:
				if str(col) in self.ignoreCols:
					inclVal = False
			if inclVal:
				outX[col] = self.useDict[col].transform( inpX[col].to_numpy().reshape(-1,1) )
		return outX


#
class AddPCA():
	
	def __init__(self, featsToUse, nComponents=1):
		self.nComponents = nComponents
		self.featsToUse = featsToUse
		
	def fit(self, inpX, y=None):
		self.pcaObj = sk.decomposition.PCA(n_components=self.nComponents)
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]
		
		#Need to scale before doing PCA
		self.scaler = StandardScaler()
		self.scaler.fit(useFrame)
		useFrame = self.scaler.transform(useFrame)

		#
		self.pcaObj.fit(useFrame)
		
		return self
	
	def transform(self, inpX):
		#Setup frame for PCA
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]
		useFrame = self.scaler.transform(useFrame)

		#Add components
		pcVals = self.pcaObj.transform(useFrame)
		colNames = ["pc_{}".format(x) for x in range(pcVals.shape[1])]
		pcFrame = pd.DataFrame(pcVals, index=useFrame.index, columns=colNames)
		
		#
		outFrame = inpX.copy()
		outFrame = outFrame.join( pcFrame )
		return outFrame


#
class AddKMeansClusters():

	def __init__(self, featsToUse, nClusters, randomState=0, idxLabel="clusterIdx", nInit=10, 
				 distPrefix="cDist_", useDistVector=True, useLabel=True, useMinDist=False, minDistLabel="clusterMinDist"):
		self.featsToUse = featsToUse
		self.nClusters = nClusters
		self.randomState = randomState
		self.idxLabel = idxLabel
		self.distPrefix = distPrefix
		self.useDistVector = useDistVector
		self.useLabel = useLabel
		self.useMinDist = useMinDist
		self.minDistLabel = minDistLabel
		self.nInit = nInit

	def fit(self, inpX, y=None):
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]

		#Need to scale before kmeans
		self.scaler = StandardScaler()
		self.scaler.fit(useFrame)
		useFrame = self.scaler.transform(useFrame)

		#
		self.kMeansObj = sk.cluster.KMeans(n_clusters=self.nClusters, random_state=self.randomState, n_init=self.nInit)
		self.kMeansObj.fit(useFrame)

		return self

	def transform(self, inpX, y=None):
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]
		useFrame = self.scaler.transform(useFrame)

		#
		outIndices = pd.Series( self.kMeansObj.predict(useFrame), name=self.idxLabel, index=useFrame.index )
		colNames = [ self.distPrefix + "{}".format(idx) for idx in range(self.kMeansObj.n_clusters) ]
		outDists = pd.DataFrame( self.kMeansObj.transform(useFrame), columns=colNames, index=useFrame.index)

		#
		outFrame = inpX.copy()
		if self.useLabel:
			outFrame = outFrame.join( outIndices )
		if self.useDistVector:
			outFrame = outFrame.join( outDists )

		if self.useMinDist:
			_temp = outFrame[ colNames ].to_numpy()
			minClusterDist = pd.Series( np.min(_temp,axis=1), index=outFrame.index, name=self.minDistLabel )
			outFrame[ self.minDistLabel ] = minClusterDist

		return outFrame
 



#
class MEncodeMultiple():

	def __init__(self, useCols, mVal=1, targCol="SalePrice"):
		self.encoders = [MEncodeSingle(useCol, mVal=mVal, targCol=targCol) for useCol in useCols]

	def fit(self, inpX, y=None):
		for enc in self.encoders:
			enc.fit(inpX, y=y)
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		for enc in self.encoders:
			outX = enc.transform(outX)
		return outX


class MEncodeSingle():
	
	def __init__(self, useCol, mVal=1, targCol="SalePrice"):
		self.useCol = useCol
		self.targCol = targCol
		self.mVal = mVal
		
	def fit(self, inpX, y=None):
		self._globalMean = inpX[self.targCol].mean()
		self._groupInfo = inpX.groupby(self.useCol)[self.targCol].aggregate(["count","mean"])
		return self
	
	def transform(self, inpX):
		outFrame = inpX.copy()
		
		#
		_tempFrame = self._groupInfo.copy()
		outName = self.useCol + "_m{}".format(int(self.mVal))
		_tempFrame[outName] = 0
		
		def _applyFunct(inpRow):
			factor = inpRow["count"] / (inpRow["count"] + self.mVal)
			groupContrib = factor*inpRow["mean"]
			globContrib = (1-factor)*self._globalMean
			inpRow[outName] = groupContrib + globContrib
			return inpRow
		
		_tempFrame = _tempFrame.apply(_applyFunct, axis=1).reset_index()[[self.useCol,outName]]
		_currKwargs = {"left_on":self.useCol,"right_on":self.useCol,"how":"left"}
		outFrame = pd.merge(outFrame, _tempFrame, **_currKwargs)
		outFrame[outName] = outFrame[outName].map(lambda x: self._globalMean if pd.isna(x) else x  )
		
		return outFrame


#
class TransformNumericalNaN():
	
	def __init__(self, **kwargs):
		self.useDtypes = ["int","float"]
		self.mappings = _getDefaultNanMappings()
		self.mappings.update(kwargs)
	
	def fit(self, inpX, inpY=None):
		""" Input is the training dataFrame; fit generally means calculating and storing the relevant means"""
		#Calculate before assigning; dont want class attrs partially overwritten in the case of an error thrown here
		meanDict = inpX.select_dtypes(self.useDtypes).mean()
		medianDict = inpX.select_dtypes(self.useDtypes).median()
		
		#
		self.meanDict, self.medianDict = meanDict, medianDict
	
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()

		#Figure out which fields have NaN values
		fieldsWithNan = outX.select_dtypes(self.useDtypes).isna().sum().gt(0)
		fieldsWithNan = fieldsWithNan.loc[ fieldsWithNan==True ].index.to_list()
		
		#
		outFrame = outX.copy()
		for field in fieldsWithNan:
			replaceNaNValsForField(outFrame, field, self.mappings.get(field,"zero"),
								   mean=self.meanDict[field], median=self.medianDict[field])
		
		return outFrame
		
def replaceNaNValsForField(inpFrame, inpField, replStrat, mean=None, median=None):
	if replStrat == "zero":
		inpFrame[inpField] = inpFrame[inpField].map( lambda x: 0 if pd.isna(x) else x )
	elif replStrat == "mean":
		inpFrame[inpField] = inpFrame[inpField].map( lambda x: mean if pd.isna(x) else x )
	elif replStrat == "median":
		inpFrame[inpField] = inpFrame[inpField].map( lambda x: median if pd.isna(x) else x  )
		
	#Slightly messy; case in which we replace with another fields value
	elif _isIter(replStrat):
		if replStrat[0] == "otherAttr":
			_replaceNaNWithOtherField(inpFrame, inpField, replStrat[1])
		else:
			raise KeyError("{} for field {}".format(replStrat,inpField))
	
	else:
		raise KeyError("{} for field {}".format(replStrat,inpField))


def _replaceNaNWithOtherField(inpFrame, inpField, replField):
	nanFrame = inpFrame.loc[ inpFrame[inpField].isna() ]
	inpFrame[inpField].loc[ nanFrame.index ] = inpFrame[replField].loc[ nanFrame.index ]

  
def _isIter(inp):
	try:
		iter(inp)
	except TypeError:
		return False
	else:
		return True

def _getDefaultNanMappings():
	outDict = dict()
	#In the train part
	outDict["MasVnrArea"] = "zero"
	outDict["GarageYrBlt"] = ["otherAttr", "YearBuilt"]
	outDict["LotFrontage"] = "median"
	
	#
	outDict["GarageCars"] = "zero"
	outDict["TotalBsmtSF"] = "zero"
	outDict["BsmtUnfSF"] = "zero"
	outDict["BsmtFinSF2"] = "zero"
	outDict["BsmtFinSF1"] = "zero"
	outDict["GarageArea"] = "zero"
	outDict["BsmtFullBath"] = "zero"
	outDict["BsmtHalfBath"] = "zero"
	
	return outDict

