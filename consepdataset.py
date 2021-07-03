import os
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

from util_hv_map import get_hv_map

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
		horizontal_map, vertical_map = get_hv_map(labels['inst_map'].astype(int), image)
		
		image, label_inst, label_type, horizontal_map, vertical_map = simpleTransform(image, labels['inst_map'], labels['type_map'], horizontal_map, vertical_map, sideLength = self.sideLength, valid = self.valid)

		label_inst = torch.from_numpy(label_inst).long()
		label_type = torch.from_numpy(label_type).long()
		image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255
		
		horizontal_map = torch.from_numpy(np.transpose(horizontal_map / 255.0, (2, 0, 1)))
		vertical_map = torch.from_numpy(np.transpose(vertical_map / 255.0, (2, 0, 1)))

		return image, label_inst, label_type, horizontal_map, vertical_map

class ConsepSimpleDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		
		horizontal_map, vertical_map = get_hv_map(labels['inst_map'].astype(int), image)
		
		image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255.0
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		
		horizontal_map = torch.from_numpy(np.transpose(horizontal_map / 255.0, (2, 0, 1)))
		vertical_map = torch.from_numpy(np.transpose(vertical_map / 255.0, (2, 0, 1)))

		return image, label_inst, label_type, horizontal_map, vertical_map
	

class ConsepSimplePadDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		
		horizontal_map, vertical_map = get_hv_map(labels['inst_map'].astype(int), image)
		
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1)))
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		
		horizontal_map = torch.from_numpy(np.transpose(horizontal_map / 255.0, (2, 0, 1)))
		vertical_map = torch.from_numpy(np.transpose(vertical_map / 255.0, (2, 0, 1)))

		m = nn.ZeroPad2d(12)

		return m(image), m(label_inst), m(label_type), m(horizontal_map - 0.5) + 0.5, m(vertical_map - 0.5) + 0.5
