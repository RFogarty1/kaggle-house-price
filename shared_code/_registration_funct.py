import contextlib

#Old code nicked from here:
#https://github.com/RFogarty1/plato_gen_basis_helpers/blob/master/gen_basis_helpers/shared/register_key_decorator.py


class RegisterKeyValDecorator():
	"""Callable class which acts as a decorator for registering a key and value to a specific dictionary
	The callable function takes (key,val,**kwargs) as arguments, where **kwargs can be set to ANY attributes of the instance (listed below). These attributes will be changed ONLY for the function call if set this way (i.e. only set temporarily)
	Attributes:
		register: (dict) The dictionary which key,val pairs will be added to
		forceKeysToCase: (str/None) If None, then keys are added to register without modification. If "upper" or "lower" they are converted to the relevant case first
		overwrite: (Bool) If True, then adding the same key twice will cause the first to be overwritten, if False then an AssertionError will be raised if the input key is already present in register
	"""

	def __init__(self, register, forceKeysToCase=None, overwrite=False):
		self.register = register
		self.forceKeysToCase = forceKeysToCase
		self.overwrite=overwrite

	def __call__(self, key, val=None, **kwargs):
		with temporarilySetInstanceAttrs(self,kwargs):
			registerKey = self._getFormattedKey(key)
			if self.overwrite is False:
				assert registerKey not in self.register.keys()
			def decoFunct(val):
				self.register[registerKey] = val
				return val #Needed for stacking decorators

			if val is None: #Allows this to be used with the @ decorator syntax
				return decoFunct
			else:
				decoFunct(val) #Allows this to be used with decoratorObj(key,thingToDecorate) syntax


	def _getFormattedKey(self, key):
		if self.forceKeysToCase == "lower":
			return key.lower()
		elif self.forceKeysToCase == "upper":
			return key.upper()
		elif self.forceKeysToCase == None:
			return key
		else:
			raise ValueError("{} is an invalid keyword for forceKeysToCase".format(self.forceKeysToCase))


@contextlib.contextmanager
def temporarilySetInstanceAttrs(instance, modKwargDict):
	origAttrs = dict()
	for key in modKwargDict:
		origAttrs[key] = getattr(instance,key)

	for key in modKwargDict:
		setattr(instance,key,modKwargDict[key])

	try:
		yield
	finally:
		for key in origAttrs:
			setattr(instance,key,origAttrs[key])	

