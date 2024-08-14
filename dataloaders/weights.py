import numpy
import torch
import torchvision
# from torchvision import transforms, datasets
from .wrapper import CacheClassLabel, MyDataset, AppendName
# from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import os
import pandas as pd
# from torchvision.io import read_image
# from torchvision.transforms import Compose, ToTensor, Normalize
import pickle
from torch.utils.data import Dataset
import logging
from .data_utils import load_transforms



def load_all_data(root_dir=None, dataset=None):
	if dataset is None or root_dir is None:
		raise ValueError('Dataset or Root Dir not defined.')
	data = []
	if dataset == 'RAFDB':
		image_dir = os.path.join(root_dir, 'Image/aligned')
		emotion_labels_path = os.path.join(root_dir, 'EmoLabel/list_patition_label.txt')
		emotion_labels = pd.read_csv(emotion_labels_path, sep=' ', names=['image_name', 'emotion_label'], header=None)
		emotion_labels['image_name'] = emotion_labels['image_name'].str.replace('.jpg', '_aligned.jpg')
		# Loop through the dataset to load images and labels
		for idx in range(len(emotion_labels)):
			image_path = os.path.join(image_dir, emotion_labels.iloc[idx, 0])
			emotion_label = emotion_labels.iloc[idx, 1] - 1
			#print(emotion_labels['emotion_label'].unique())
			data.append([image_path, emotion_label])
	
	elif dataset == 'BP4D':
	
		image_dir = root_dir + '/images_crop'
		
		AU_labels_path = root_dir + '/aus_bp4d_occurrence.pkl'
		au_list = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
		
		with open(AU_labels_path, 'rb') as f:
			files_labels = pickle.load(f)
			
		images = list(files_labels.keys())
		au_labels = list(files_labels.values())
		data = [(image_dir + "/" + images[i] + ".jpg", au_labels[i]) for i in range(len(images)) if os.path.exists(image_dir + "/" + images[i] + ".jpg")]
		
	elif dataset == 'AffWild2':
		images_dir = os.path.join(root_dir, "images")
		annotations_dir = os.path.join(root_dir, 'annotations/annotations/VA_Estimation_Challenge/Train_Set/')
		
		for folder_name in os.listdir(images_dir):
			images_folder_path = os.path.join(images_dir, folder_name)
			annotation_file = f'{folder_name}.txt'
			annotation_path = os.path.join(annotations_dir, annotation_file)
			
			if not os.path.exists(images_folder_path) or not os.path.exists(annotation_path):
				print("continuing")
				continue
			
			# Read arousal and valence values
			annotations = pd.read_csv(annotation_path)
			
			for i, row in annotations.iterrows():
				if not os.path.exists(os.path.join(images_folder_path, f'{(i + 1):05}.jpg')):
					continue
				image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
				data.append((image_path, [row['valence'], row['arousal']]))
		
	
	return numpy.asarray(data).reshape((len(data), 2))


class AUDataset(Dataset):
	def __init__(self, data, transform=None):
		self.dataX = data[:, 0]
		self.dataY = data[:, 1]
		self.transform = transform
	
	def __len__(self):
		return len(self.dataX)
	
	def __getitem__(self, idx):
		image = Image.open(self.dataX[idx]).convert('RGB')
		label = torch.tensor(self.dataY[idx], dtype=torch.int64)
		if self.transform:
			image = self.transform(image)
		
		return image, label

class AVDataset(Dataset):
	def __init__(self, data, transform=None):
		self.dataX = data[:, 0]
		self.dataY = numpy.asarray(data[:, 1])
		self.transform = transform

	def __len__(self):
		return len(self.dataX)
	
	def __getitem__(self, idx):
		image = Image.open(self.dataX[idx]).convert('RGB')
		label = torch.tensor([self.dataY[idx][0].astype('float32'), self.dataY[idx][1].astype('float32')], dtype=torch.float32)
		if self.transform:
			image = self.transform(image)
		
		return image, label
class FERDataset(Dataset):
	def __init__(self, data, transform=None):
		self.dataX = data[:, 0]
		self.dataY = numpy.asarray(data[:, 1]).astype('int64')
		self.transform = transform
	
	def __len__(self):
		return len(self.dataX)
	
	def __getitem__(self, idx):
		image = Image.open(self.dataX[idx]).convert('RGB')
		label = torch.tensor(self.dataY[idx], dtype=torch.int64)
		# label = torch.nn.functional.one_hot(torch.tensor(self.dataY[idx].astype('float32'), dtype=torch.float32))
		
		if self.transform:
			image = self.transform(image)
		
		return image, label
	
def extract_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    return np.array(labels)

def calculate_class_weights(labels):

    # Calculate the frequency of each class
    class_frequencies = np.sum(labels, axis=0)
    
    # Avoid division by zero by adding a small value to frequencies
    class_frequencies = np.clip(class_frequencies, 1, None)
    
    # Compute the inverse of these frequencies
    inverse_frequencies = 1.0 / class_frequencies
	#weights = [(float(i)/sum(inverse_frequencies)) * len(inverse_frequencies) for i in inverse_frequencies]

    # Normalize the weights
    normalized_weights = inverse_frequencies * len(class_frequencies) / np.sum(inverse_frequencies)
    
    return normalized_weights

def calc_weights(classes):
	
	
	for key in classes:
		
		if key == 'FER':
			rafdbdata = './data/Rafdb'
			FERdata = load_all_data(root_dir=rafdbdata, dataset='RAFDB')
			labels = extract_labels(FERdata)
			weights = calculate_class_weights(labels)
		
		elif key == 'AV':
			affwild2 = './data/affwild2'
			AVdata = load_all_data(root_dir=affwild2, dataset='AffWild2')
			labels = extract_labels(AVdata)
			weights = calculate_class_weights(labels)

		elif key == 'AU':
			bp4d = './data/bp4d'
			AUdata = load_all_data(root_dir=bp4d, dataset='BP4D')
			labels = extract_labels(AUdata)
			weights = calculate_class_weights(labels)

		
	
	return weights



