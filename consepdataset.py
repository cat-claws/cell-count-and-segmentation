import os
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

from util_hv_map import get_hv_map, gen_targets

def extendLabels(force = False):
	""" get horizontal/vertical maps before loading"""
	
	directories = ['CoNSeP/Train', 'CoNSeP/Test']
	setnames = ['train', 'test']
	for i in range(len(directories)):
		for index in range(len(os.listdir(os.path.join(directories[i], 'Labels')))):
			labels = scipy.io.loadmat(os.path.join(directories[i], 'Labels', setnames[i] + f'_{index + 1}.mat'))
			image = np.array(Image.open(os.path.join(directories[i], 'Images', setnames[i] + f'_{index + 1}.png')))[:,:,:3]

			if ('hori_map' not in labels or 'vert_map' not in labels) or force:
				map_dict = gen_targets(labels['inst_map'].astype(int), [1000, 1000])
				labels['hori_map'] = map_dict['h_map']
				labels['vert_map'] = map_dict['v_map']
				scipy.io.savemat(os.path.join(directories[i], 'Labels', setnames[i] + f'_{index + 1}.mat'), labels)

extendLabels()			

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
	def __init__(self, train = False, test = False, valid = False, sideLength = 256, num = 200, combine_classes = True):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
# 		assert train != test and train != valid
		self.sideLength = sideLength
		self.num = num
		self.valid = valid
		self.combine_classes = combine_classes

	def __len__(self):
		return self.num #if not test else len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		index = index % len(os.listdir(os.path.join(self.directory, 'Images')))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		
		image, label_inst, label_type, hori_map, vert_map = simpleTransform(image, labels['inst_map'], labels['type_map'], labels['hori_map'], labels['vert_map'], sideLength = self.sideLength, valid = self.valid)

		label_inst = torch.from_numpy(label_inst).long()
		label_type = torch.from_numpy(label_type).long()
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(hori_map).unsqueeze(0).float()
		vert_map = torch.from_numpy(vert_map).unsqueeze(0).float()

		return image, label_inst, label_type, hori_map, vert_map

class ConsepSimpleDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False, combine_classes = True):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test
		self.combine_classes = combine_classes

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
			
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(labels['hori_map']).float()
		vert_map = torch.from_numpy(labels['vert_map']).float()

		return image, label_inst, label_type, hori_map, vert_map
	

class ConsepSimplePadDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False, combine_classes = True):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train != test
		self.combine_classes = combine_classes

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
			
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(np.transpose(labels['hori_map'] / 255.0, (2, 0, 1))).float()
		vert_map = torch.from_numpy(np.transpose(labels['vert_map'] / 255.0, (2, 0, 1))).float()

		m = nn.ZeroPad2d(12)

		return m(image), m(label_inst), m(label_type), m(hori_map - 0.5) + 0.5, m(vert_map - 0.5) + 0.5
