
import numpy as np

import sklearn as sk
import sklearn.base
import sklearn.ensemble

import xgboost as xgb

class _ScoreMixin():

	def score(self, inpX, inpY=None, sample_weight=None):
		predVals = self.predict(inpX)
		actVals = inpX[self.targCol]
		outScore = _getRMSE_ofLogged(predVals, actVals)
		return outScore


def _getRMSE_ofLogged(predVals, actVals):
	logPred, logAct = np.log(predVals), np.log(actVals)
	squaredError = (logPred-logAct)**2
	return np.sqrt( np.mean(squaredError) )

class RandomForestWrapper( sk.base.BaseEstimator, _ScoreMixin ):

	def __init__(self, feats, targCol="SalePrice", trainPipe=None, rfKwargs=None, logTarget=False):
		self.feats = feats
		self.targCol = targCol
		self.trainPipe = trainPipe
		self.rfKwargs = rfKwargs #Needed for the clone method
		self.logTarget = logTarget 

		_currKwargs = dict() if rfKwargs is None else rfKwargs
		self.regressor = sk.ensemble.RandomForestRegressor(**_currKwargs)

	def fit(self, inpX, y=None):
		useX = inpX.copy()

		#Modify training data if required
		if self.trainPipe is not None:
			useX = self.trainPipe.fit_transform(useX)

		#Fit a random forest classifier
		useTrain = useX[self.feats]
		useTarg = inpX[self.targCol] if self.logTarget is False else np.log(inpX[self.targCol])

		self.regressor.fit( useTrain.to_numpy(), useTarg )

		return self

	def predict(self, inpX):
		useX = inpX.copy()

		#Apply processing
		if self.trainPipe is not None:
			useX = self.trainPipe.transform(useX)

		#
		useX = useX[self.feats]
		output = self.regressor.predict(useX.to_numpy())

		if self.logTarget:
			output = np.exp(output)

		return output




#Largely the same as RF wrapper. 
class XGBoostWrapper( sk.base.BaseEstimator, _ScoreMixin ):

	def __init__(self, feats, targCol="SalePrice", trainPipe=None, xgbKwargs=None, logTarget=False):
		self.feats = feats
		self.targCol = targCol
		self.trainPipe = trainPipe
		self.xgbKwargs = xgbKwargs #Needed for the clone method
		self.logTarget = logTarget 

		_currKwargs = dict() if xgbKwargs is None else xgbKwargs
		self.regressor = xgb.XGBRegressor(**_currKwargs)

	def fit(self, inpX, y=None):
		useX = inpX.copy()

		#Modify training data if required
		if self.trainPipe is not None:
			useX = self.trainPipe.fit_transform(useX)

		#Fit a random forest classifier
		useTrain = useX[self.feats]
		useTarg = inpX[self.targCol] if self.logTarget is False else np.log(inpX[self.targCol])

		self.regressor.fit( useTrain.to_numpy(), useTarg )

		return self

	def predict(self, inpX):
		useX = inpX.copy()

		#Apply processing
		if self.trainPipe is not None:
			useX = self.trainPipe.transform(useX)

		#
		useX = useX[self.feats]
		output = self.regressor.predict(useX.to_numpy())

		if self.logTarget:
			output = np.exp(output)

		return output

