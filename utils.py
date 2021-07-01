import torch

def labelRatios(iterator, index, n_classes = 8):
	ratio = torch.ones(n_classes) # normally, it should start from zero, though we start with one to avoid 'divide by zero'
	for batch in iterator:
		for i in range(n_classes):
			ratio[i] += torch.sum(batch[index] == i) # this is a super slow method, though it does not hurt so far
	return ratio

def contour(inst_map):
	borderExcluded = torch.nn.ZeroPad2d(1)(torch.ones((inst_map.shape[-2] - 2, inst_map.shape[-1] - 2))).bool()
	c = torch.zeros_like(inst_map)
	for i in (-1, 1):
		for j in (-1, 1):
			c.masked_fill_(~torch.eq(inst_map, torch.roll(inst_map, shifts=(i, j), dims=(-2, -1))), 1)
	return torch.logical_and(c.bool(), borderExcluded)
