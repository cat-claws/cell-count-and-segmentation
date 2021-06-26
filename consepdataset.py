import os
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

def simpleTransform(*target, sideLength):
	H, W = np.array(target[0].shape[:2]) - sideLength
	assert H * W * (H + W) > 0
	x = np.random.randint(H)
	y = np.random.randint(W)

	target = [t[x:x+sideLength, y:y+sideLength] for t in target]

	if random.random() > 0.5:
		target = [np.flip(t, 0) for t in target]
	if random.random() > 0.5:
		target = [np.flip(t, 1) for t in target]

	return tuple([t.copy() for t in target])

class ConsepSimpleTransformDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False, sideLength = 512):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test
		self.sideLength = sideLength

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image, label_inst, label_type = simpleTransform(image, labels['inst_map'], labels['type_map'], sideLength=self.sideLength)

		label_inst = torch.from_numpy(label_inst).long()
		label_type = torch.from_numpy(label_type).long()
		image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255

		return image, label_inst, label_type

class ConsepSimpleDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255

		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()

		return image, label_inst, label_type
	

class ConsepSimplePadDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255

		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()

		m = nn.ZeroPad2d(12)

		return m(image), m(label_inst), m(label_type)
