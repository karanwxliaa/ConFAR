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


#CLASSES FOR DATASETS
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
    



#LOAD FUNCTIONS

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
            #print("Appending ",[image_path, emotion_label])
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
    
    return numpy.asarray(data, dtype=object).reshape((len(data), 2))


def load_all_data_af2(root_dir=None, task=None):
    if task is None or root_dir is None:
        raise ValueError('Dataset or Root Dir not defined.')
    
    
    if task == 'FER':
        data = []
        images_dir = os.path.join(root_dir, "images")
        annotations_dir = os.path.join(root_dir, 'annotations/annotations/EXPR_Classification_Challenge/Train_Set/')
        
        for folder_name in os.listdir(images_dir):
            images_folder_path = os.path.join(images_dir, folder_name)
            #annotation_file = f'{folder_name}.txt'
            annotation_path = os.path.join(annotations_dir, f'{folder_name}.txt')
            
            if not os.path.exists(images_folder_path) or not os.path.exists(annotation_path):
                #print(" folder ", annotation_path, " doesnt exist!")
                continue
            
            # Adjusted read_csv call to correctly handle the file format
            annotations = pd.read_csv(annotation_path, header=None, skiprows=1, usecols=[0])

            for i, row in annotations.iterrows():
                if not os.path.exists(os.path.join(images_folder_path, f'{(i + 1):05}.jpg')):
                    continue
                image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
                # Assuming the label is the first (and only) element in the row
                label = row[0]
                if label == -1 or label == 7:
                    label = 0  #CHANGE THIS TO CONTINUE INSTEAD LATER !!!
                    data.append((image_path, label))
                else:
                    #print("Appending ",[image_path, label])
                    data.append((image_path, label))



    elif task == 'AU':
        data = []
        images_dir = os.path.join(root_dir, "images")
        annotations_dir = os.path.join(root_dir, 'annotations/annotations/AU_Detection_Challenge/Train_Set/')
        
        for folder_name in os.listdir(images_dir):
            images_folder_path = os.path.join(images_dir, folder_name)
            annotation_file = f'{folder_name}.txt'
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            if not os.path.exists(images_folder_path) or not os.path.exists(annotation_path):
                continue
            
            # Read arousal and valence values
            annotations = pd.read_csv(annotation_path)
            
    
        for i, row in annotations.iterrows():
            if not os.path.exists(os.path.join(images_folder_path, f'{(i + 1):05}.jpg')):
                continue
            image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
            data.append((image_path, row))

        
    elif task == 'AV':
        data = []
        images_dir = os.path.join(root_dir, "images")
        annotations_dir = os.path.join(root_dir, 'annotations/annotations/VA_Estimation_Challenge/Train_Set/')
        
        for folder_name in os.listdir(images_dir):
            images_folder_path = os.path.join(images_dir, folder_name)
            annotation_file = f'{folder_name}.txt'
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            if not os.path.exists(images_folder_path) or not os.path.exists(annotation_path):
                continue
            
            # Read arousal and valence values
            annotations = pd.read_csv(annotation_path)
            
            for i, row in annotations.iterrows():
                if not os.path.exists(os.path.join(images_folder_path, f'{(i + 1):05}.jpg')):
                    continue
                image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
                data.append((image_path, [row['valence'], row['arousal']]))
    
    return numpy.asarray(data, dtype=object).reshape((len(data), 2))



def load_all_data_ub(root_dir, max_images=10000):
    data = []
    images_dir = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, 'annotations/annotations')

    # Directories for each task
    au_dir = os.path.join(annotations_dir, 'AU_Detection_Challenge/Train_Set/')
    expr_dir = os.path.join(annotations_dir, 'EXPR_Classification_Challenge/Train_Set/')
    av_dir = os.path.join(annotations_dir, 'VA_Estimation_Challenge/Train_Set/')

    for folder_name in os.listdir(images_dir):

        images_folder_path = os.path.join(images_dir, folder_name)
        au_annotation_path = os.path.join(au_dir, f'{folder_name}.txt')
        expr_annotation_path = os.path.join(expr_dir, f'{folder_name}.txt')
        av_annotation_path = os.path.join(av_dir, f'{folder_name}.txt')
        
        # Ensure all paths exist
        if not os.path.exists(images_folder_path) or not all(os.path.exists(path) for path in [au_annotation_path, expr_annotation_path, av_annotation_path]):
            continue
        
        # Read annotations
        au_annotations = pd.read_csv(au_annotation_path, skiprows=1, header=None)
        expr_annotations = pd.read_csv(expr_annotation_path, header=None, skiprows=1, usecols=[0])
        av_annotations = pd.read_csv(av_annotation_path)

        for i, row in av_annotations.iterrows():

            image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
            if not os.path.exists(image_path):
                continue

            # Load labels
            au_labels = au_annotations.iloc[i].values
            expr_label = int(expr_annotations.iloc[i][0])
            if expr_label == -1 or expr_label == 7:
                expr_label = 0
            av_label = [row['valence'], row['arousal']]
            target = {"AU": au_labels, "FER": expr_label, "AV": av_label}

            data.append((image_path, target))

    return np.asarray(data, dtype=object).reshape((len(data), 2))


def load_10k_data_ub(root_dir, max_images=10000):
    data = []
    images_dir = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, 'annotations/annotations')

    # Directories for each task
    au_dir = os.path.join(annotations_dir, 'AU_Detection_Challenge/Train_Set/')
    expr_dir = os.path.join(annotations_dir, 'EXPR_Classification_Challenge/Train_Set/')
    av_dir = os.path.join(annotations_dir, 'VA_Estimation_Challenge/Train_Set/')

    image_count = 0  # Initialize image counter

    for folder_name in os.listdir(images_dir):
        if image_count >= max_images:
            break  # Exit the loop if max_images limit is reached

        images_folder_path = os.path.join(images_dir, folder_name)
        au_annotation_path = os.path.join(au_dir, f'{folder_name}.txt')
        expr_annotation_path = os.path.join(expr_dir, f'{folder_name}.txt')
        av_annotation_path = os.path.join(av_dir, f'{folder_name}.txt')
        
        # Ensure all paths exist
        if not os.path.exists(images_folder_path) or not all(os.path.exists(path) for path in [au_annotation_path, expr_annotation_path, av_annotation_path]):
            continue
        
        # Read annotations
        au_annotations = pd.read_csv(au_annotation_path, skiprows=1, header=None)
        expr_annotations = pd.read_csv(expr_annotation_path, header=None, skiprows=1, usecols=[0])
        av_annotations = pd.read_csv(av_annotation_path)

        for i, row in av_annotations.iterrows():
            if image_count >= max_images:
                break  # Exit the loop if max_images limit is reached

            image_path = os.path.join(images_folder_path, f'{(i + 1):05}.jpg')
            if not os.path.exists(image_path):
                continue

            # Load labels
            au_labels = au_annotations.iloc[i].values
            expr_label = int(expr_annotations.iloc[i][0])
            if expr_label == -1 or expr_label == 7:
                expr_label = 0
            av_label = [row['valence'], row['arousal']]
            target = {"AU": au_labels, "FER": expr_label, "AV": av_label}

            data.append((image_path, target))
            image_count += 1  # Increment image counter

    return np.asarray(data, dtype=object).reshape((len(data), 2))



class UnifiedDataset(Dataset):
    def __init__(self, data, task, transform=None):
        self.dataX = data[:, 0]  # Image paths
        self.dataY = data[:, 1]  # Combined labels, ensure this is correctly formatted in load_data
        self.transform = transform
        self.task = task
    
    def __len__(self):
        return len(self.dataX)
    
    def __getitem__(self, idx):
        image_path = self.dataX[idx]
        target = self.dataY[idx][self.task]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Initialize a padded target tensor of length 12 (maximum output space)
        padded_target = torch.zeros(12, dtype=torch.float32)
        
        if self.task == "AU":
            # For AU, assume target is already the correct length (12)
            label = torch.tensor(target, dtype=torch.float32)  # Assuming AU labels are binary [0, 1]
        elif self.task == "AV":
            # For AV, place the 2 values at the start of the padded tensor
            label = torch.tensor([target[0].astype('float32'), target[1].astype('float32')], dtype=torch.float32)
            padded_target[:2] = label
            label = padded_target
        elif self.task == "FER":
            # For FER, place the single label at the start of the padded tensor
            padded_target[0] = target
            label = padded_target.long()  # Convert back to long for classification
            
        return image, label


#CONFAR LOADING FUNCTIONS
    
def load_ConFAR_UB(image_size, aug):
    train_transform, val_transform = load_transforms(image_size=image_size, aug=aug)

    affwild2 = './data/affwild2'
    
    train_datasets = {}  # 1: RAF-DB train 2: bp4d Tran and 3: Affwild2 Train
    val_datasets = {}
    
    task_output_space =  {"FER": 7,"AV": 2,"AU": 12}
    

    #AllData = load_all_data_ub(root_dir=affwild2)
    AllData = load_10k_data_ub(root_dir=affwild2,max_images=306080) #total 306235;  80% of 306080 is 244864 which is a direct multiple of 128 for NR)
    



    train_dataset, val_dataset = train_test_split(AllData, test_size=0.2, random_state=42)
    
    train_dataset_fer = UnifiedDataset(data=train_dataset, transform=train_transform,task = "FER")
    val_dataset_fer = UnifiedDataset(data=val_dataset, transform=val_transform,task = "FER")

    train_dataset_au = UnifiedDataset(data=train_dataset, transform=train_transform,task = "AU")
    val_dataset_au = UnifiedDataset(data=val_dataset, transform=val_transform,task = "AU")

    train_dataset_av = UnifiedDataset(data=train_dataset, transform=train_transform,task = "AV")
    val_dataset_av = UnifiedDataset(data=val_dataset, transform=val_transform,task = "AV")
                
    

    logging.info("Loaded AffWild2 for ALLDATA")
    print("Loaded AffWild2 for ALLDATA")

        
    train_datasets["FER"] = AppendName(train_dataset_fer, "FER")
    val_datasets["FER"] = AppendName(val_dataset_fer, "FER")

    train_datasets["AU"] = AppendName(train_dataset_au, "AU")
    val_datasets["AU"] = AppendName(val_dataset_au, "AU")

    train_datasets["AV"] = AppendName(train_dataset_av, "AV")
    val_datasets["AV"] = AppendName(val_dataset_av, "AV")

    return train_datasets, val_datasets, task_output_space
    

    
    
def load_ConFAR(image_size, aug):
    train_transform, val_transform = load_transforms(image_size=image_size, aug=aug)
    
    rafdbdata = './data/Rafdb'
    bp4d = './data/bp4d'
    affwild2 = './data/affwild2'
    
    train_datasets = {}  # 1: RAF-DB train 2: bp4d Tran and 3: Affwild2 Train
    val_datasets = {}
    
    task_output_space = {"AV": 2,"AU": 12, "FER": 7}
    
    
    for key in task_output_space:
        if key == 'FER':
            FERdata = load_all_data(root_dir=rafdbdata, dataset='RAFDB')
            train_dataset, val_dataset = train_test_split(FERdata, test_size=0.2, random_state=42)
            train_dataset = FERDataset(data=train_dataset, transform=train_transform)
            val_dataset = FERDataset(data=val_dataset, transform=val_transform)
            
            logging.info("Loaded RAFDB for FER")
            print("Loaded RAFDB paths for FER")
        
        elif key == 'AV':
            AVdata = load_all_data(root_dir=affwild2, dataset='AffWild2')
            train_dataset, val_dataset = train_test_split(AVdata, test_size=0.2, random_state=42)
            train_dataset = AVDataset(data=train_dataset, transform=train_transform)
            val_dataset = AVDataset(data=val_dataset, transform=val_transform)
            
            logging.info("Loaded Affwild2 for AV")
            print("Loaded Affwild2 paths for AV")
        
        elif key == 'AU':
            AUdata = load_all_data(root_dir=bp4d, dataset='BP4D')
            train_dataset, val_dataset = train_test_split(AUdata, test_size=0.2, random_state=42)
            train_dataset = AUDataset(data=train_dataset, transform=train_transform)
            val_dataset = AUDataset(data=val_dataset, transform=val_transform)
            
            logging.info("Loaded BP4D for AU")
            print("Loaded BP4D paths for AU")
        
        train_datasets[key] = AppendName(train_dataset, key)
        val_datasets[key] = AppendName(val_dataset, key)
    
    return train_datasets, val_datasets, task_output_space