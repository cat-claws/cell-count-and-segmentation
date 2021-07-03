import os
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

def simpleTransform(*target, sideLength, valid = False):
	H, W = np.array(target[0].shape[:2])
	x = np.random.randint(H * 0.7, H - sideLength) if valid else np.random.randint(H * 0.7 - sideLength)
	y = np.random.randint(W - sideLength)
	assert H * W * (H + W) > 0

	target = [t[x:x+sideLength, y:y+sideLength] for t in target]

	if np.random.random() > 0.5:
		target = [np.flip(t, 0) for t in target]
	if np.random.random() > 0.5:
		target = [np.flip(t, 1) for t in target]

	return tuple([t.copy() for t in target])

class ConsepSimpleTransformDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False, valid = False, sideLength = 256, num = 200):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
# 		assert train != test and train != valid
		self.sideLength = sideLength
		self.num = num
		self.valid = valid

	def __len__(self):
		return self.num #if not test else len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		index = index % len(os.listdir(os.path.join(self.directory, 'Images')))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image, label_inst, label_type = simpleTransform(image, labels['inst_map'], labels['type_map'], sideLength = self.sideLength, valid = self.valid)

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
