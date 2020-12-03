from .models import *

def get_model(name, num=None):
	if num==None:
		assert name in dlmodels[suffix[name.split(".")[1]]]
		return work+suffix[name.split(".")[1]] + "/" + name
	else:
		assert 0 < num < len(dlmodels[name+"-model"])
		return work + name + "-model/" + dlmodels[name+"-model"][num]