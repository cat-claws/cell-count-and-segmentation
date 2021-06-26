import torch

def labelRatios(iterator, index, n_classes = 8):
	ratio = torch.ones(n_classes) # normally, it should start from zero, though we start with one to avoid 'divide by zero'
	for batch in iterator:
		for i in range(n_classes):
			ratio[i] += torch.sum(batch[index] == i) # this is a super slow method, though it does not hurt so far
	return ratio
