import os
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

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
