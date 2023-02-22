
import knncmi




def recursivelyAddFeatsWithHighestCMI(inpFrame, targKey, stopAfterN=10, kNebs=3):
	#Initialize all
	newFeats = inpFrame.columns.to_list()
	newFeats.pop(newFeats.index(targKey))
	useFeats = list()
	useMI = list()

#	while len(newFeats) > leaveN:
	while (len(useFeats) < stopAfterN) and len(newFeats)>0:
		newFeat, newMI = _getFeatWithHighestMI(inpFrame, newFeats, useFeats, kNebs=kNebs)
		useFeats.append(newFeat)
		useMI.append(newMI)
		newFeats.pop( newFeats.index(newFeat) )


	useFeatVsMI = [ [feat,mi] for feat,mi in zip(useFeats, useMI) ]

	return useFeatVsMI
	
def _getFeatWithHighestMI(inpFrame, inpFeats, outFeats, kNebs=3):
	outFeatVsMI = list()
	for feat in inpFeats:
		currMI = knncmi.cmi([feat], ["SalePrice"], outFeats, kNebs, inpFrame)
		outFeatVsMI.append( [feat,currMI] )

	outVals = sorted(outFeatVsMI, key=lambda x:x[1], reverse=True)
	return outVals[0]


