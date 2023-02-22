
import pandas as pd
import _registration_funct as _regFunctHelp


_FEATURE_DICT = dict()

regAddFeature = _regFunctHelp.RegisterKeyValDecorator(_FEATURE_DICT)

def addFeatToFrame(inpFrame, featKey):
	_FEATURE_DICT[featKey](inpFrame)

@regAddFeature("BsmtFractUnfurnished")
def _unused(inpFrame):
	inpFrame["BsmtFractUnfurnished"] = inpFrame["BsmtUnfSF"]/inpFrame["TotalBsmtSF"]
	inpFrame["BsmtFractUnfurnished"] = inpFrame["BsmtFractUnfurnished"].map(lambda x: 1 if pd.isna(x) else x)

@regAddFeature("GarageAreaTimesFinish")
def _unused(inpFrame):
	inpFrame["GarageAreaTimesFinish"] = inpFrame["GarageArea"]*inpFrame["GarageFinish"]

@regAddFeature("NumbBsmtBath")
def _unused(inpFrame):
	_currFeats = ["BsmtFullBath", "BsmtHalfBath"]
	inpFrame["NumbBsmtBath"] = inpFrame[_currFeats].sum(axis=1)

@regAddFeature("NumbBath")
def _unused(inpFrame):
	_currFeats = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
	inpFrame["NumbBath"] = inpFrame[_currFeats].sum(axis=1)

@regAddFeature("OverallQualTimesCond")
def _unused(inpFrame):
	inpFrame["OverallQualTimesCond"] = inpFrame["OverallQual"]*inpFrame["OverallCond"]

@regAddFeature("PoolQualTimesCond")
def _unused(inpFrame):
	inpFrame["PoolQualTimesCond"] = inpFrame["PoolQC"]*inpFrame["PoolArea"]

@regAddFeature("PorchLikeArea")
def _unused(inpFrame):
	_currFeats = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
	inpFrame["PorchLikeArea"] = inpFrame[_currFeats].sum(axis=1)

@regAddFeature("Spaciousness")
def _unused(inpFrame):
	inpFrame["Spaciousness"] = (inpFrame["1stFlrSF"] + inpFrame["2ndFlrSF"]) / inpFrame["TotRmsAbvGrd"]

@regAddFeature("TotalSFLiv")
def _unused(inpFrame):
	inpFrame["TotalSFLiv"] = inpFrame["TotalBsmtSF"] + inpFrame["1stFlrSF"] + inpFrame["2ndFlrSF"]

@regAddFeature("TotalSFLivOverLotArea")
def _unused(inpFrame):
	addFeatToFrame(inpFrame, "TotalSFLiv")
	inpFrame["TotalSFLivOverLotArea"] = inpFrame["TotalSFLiv"] / inpFrame["LotArea"]

@regAddFeature("YearSold_Fract")
def _unused(inpFrame):
	inpFrame["YearSold_Fract"] = inpFrame["YrSold"] + (inpFrame["MoSold"]/12)


#These really need to be pre-factor
@regAddFeature("FenceQual")
def _unused(inpFrame):
	mapDict = {"MnWw":1, "GdWo":2, "MnPrv":1, "GdPrv":2}
	inpFrame["FenceQual"] = inpFrame["Fence"].map( lambda x: 0 if pd.isna(x) else mapDict[x] )

@regAddFeature("NumbStoreys_fromHouseStyle")
def _unused(inpFrame):
	mapDict = {"1Story":1, "1.5Fin":1.5, "1.5Unf":1.5, "2Story":2, "2.5Fin":2.5, "2.5Unf":2.5,
	           "SFoyer":2, "SLvl":2}
	inpFrame["NumbStoreys_fromHouseStyle"] = inpFrame["HouseStyle"].map( lambda x:mapDict[x] )

