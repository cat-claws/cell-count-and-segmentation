import torch

import numpy as np
from scipy import ndimage as ndi

def labelRatios(iterator, index, n_classes = 8):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	ratio = torch.ones(n_classes).to(device) # normally, it should start from zero, though we start with one to avoid 'divide by zero'
	for batch in iterator:
		for i in range(n_classes):
			ratio[i] += torch.sum(batch[index] == i) # this is a super slow method, though it does not hurt so far
	return ratio

def getEdges(inst_map): # unluckily, we must have this compatible with torch
	borderExcluded = torch.nn.ZeroPad2d(1)(torch.ones((inst_map.shape[-2] - 2, inst_map.shape[-1] - 2))).bool().to(inst_map.device)
	c = torch.zeros_like(inst_map).to(inst_map.device)
	for i in (-1, 1):
		for j in (-1, 1):
			c.masked_fill_(~torch.eq(inst_map, torch.roll(inst_map, shifts=(i, j), dims=(-2, -1))), 1)
	return torch.logical_and(c.bool(), borderExcluded)


def getDistanceMap(inst_map):
	binaries = (np.unique(my_inst_map)[1:] == my_inst_map[...,None]).transpose(2, 0, 1)
	layers = []
	for i in range(len(binaries)):
		layers.append(ndi.distance_transform_edt(binaries[i]))
	return np.max(np.array(layers[1:]), 0)

def scale_range(array, min = -1., max = 1.):
	array += -(np.min(array))
	array /= np.max(array) / (max - min)
	array += min
	return array

def getHVMap(inst_map):
	mesh = np.mgrid[0:inst_map.shape[0], 0:inst_map.shape[1]].astype(float)
	for k in np.unique(inst_map)[1:]:
		selected = inst_map == k
		mesh[0][selected] = scale_range(mesh[0][selected])
		mesh[1][selected] = scale_range(mesh[1][selected])
	return np.flip(mesh * (inst_map > 0), 0)


import re
from matplotlib import pyplot as plt


def viewing(directory):
	with open(directory, 'r') as f:
		text = f.read()

	loss = {}
	lossRegex = re.compile(r'Train Loss: (\d\.\d\d\d)')
	loss['train'] = [float(x) for x in lossRegex.findall(text)]
	lossRegex = re.compile(r'Val\. Loss: (\d\.\d\d\d)')
	loss['valid'] = [float(x) for x in lossRegex.findall(text)]
	print(loss)

	acc = {}
	accRegex = re.compile(r'Train Acc: (\d\d\.\d\d)')
	acc['train'] = [float(x) for x in accRegex.findall(text)]
	accRegex = re.compile(r'Val\. Acc: (\d\d\.\d\d)')
	acc['valid'] = [float(x) for x in accRegex.findall(text)]
	print(acc)

	plt.plot(loss['train'])
	plt.plot(loss['valid'])
	plt.show()

	plt.plot(acc['train'])
	plt.plot(acc['valid'])
	plt.show()

	return loss, acc
