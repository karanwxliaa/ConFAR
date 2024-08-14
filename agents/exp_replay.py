from __future__ import print_function
import torch
import torch.nn as nn
import random
import time
import sys
import torch.utils.data as data
import torch.nn.functional as F
from scipy.stats import pearsonr
import math
# from utils import predict, predict_gen
from torch.utils.data import ConcatDataset, DataLoader
import warnings
from collections import OrderedDict
import numpy as np

warnings.filterwarnings("ignore")
import torch
import numpy as np
from importlib import import_module
from .default import NormalNN
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage
from .LGR import lgr_utils
from .LGR.encoder import Classifier
from .LGR.autoencoder_latent import AutoEncoderLatent
from .default import NormalNN




class LGR_test(NormalNN):
    def __init__(self, agent_config):
        super(LGR, self).__init__(agent_config)
        #image_size, image_channels, classes, device, memory_size=200, latent_dim=100):
        self.image_size = [agent_config["img_size"],agent_config["img_size"]]
        self.memory_size = 128
        self.latent_dim = 100
        
        self.classes = 21
        
        self.image_channels = 3

        self.task_memory = {}  # To store latent vectors for replay

    def learn_batch(self, train_loader, val_loader, task_no):

        # Initialize AutoEncoder and Classifier with the provided configurations
        self.autoencoder = AutoEncoderLatent(latent_size=self.image_size[0] * self.image_size[1] * self.image_channels,
                                             classes=self.classes, z_dim=self.latent_dim)
        self.classifier = Classifier(image_size=self.image_size, image_channels=self.image_channels,
                                     classes=self.classes)
        
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

        # Train Autoencoder
        self._train_autoencoder(train_loader)
        
        # Generate and store latent representations for current task
        self._update_memory(train_loader, task_no)
        
        # Prepare combined DataLoader for current task and replayed data
        combined_loader = self._prepare_combined_loader(train_loader, task_no)

        super(LGR, self).learn_batch(combined_loader, val_loader,task_no)

   

    def _train_autoencoder(self, train_loader):
        self.autoencoder.train()
        for images, _,task in train_loader:
            images = images.to(self.device)
            self.autoencoder_optimizer.zero_grad()
            recon, _, mu, logvar = self.autoencoder(images, full=True)
            loss = self.autoencoder.loss_function(recon, images, mu=mu, logvar=logvar)[0]
            loss.backward()
            self.autoencoder_optimizer.step()

    def _update_memory(self, train_loader, task_no):
        latent_vectors = []
        for images, _ in train_loader:
            images = images.to(self.device)
            _, _, mu, _ = self.autoencoder(images, full=True)
            latent_vectors.append(mu.detach().cpu())
        latent_vectors = torch.cat(latent_vectors)[:self.memory_size]  # Limit the number of memories
        self.task_memory[task_no] = latent_vectors

    def _prepare_combined_loader(self, train_loader, task_no):
        # Generate replay data
        replay_datasets = []
        for past_task, latent_vectors in self.task_memory.items():
            if past_task == task_no:
                continue  # Skip current task
            latent_vectors = latent_vectors.to(self.device)
            replay_images = self.autoencoder.decode(latent_vectors).detach().cpu()
            replay_datasets.append(lgr_utils.TensorDataset(replay_images, torch.zeros(replay_images.size(0))))
        
        combined_dataset = torch.utils.data.ConcatDataset([train_loader.dataset] + replay_datasets)
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True)
        return combined_loader

    def _train_classifier(self, combined_loader):
        self.classifier.train()
        for images, labels in combined_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.classifier_optimizer.zero_grad()
            output = self.classifier(images)
            loss = torch.nn.functional.cross_entropy(output, labels.long())
            loss.backward()
            self.classifier_optimizer.step()
    


class BatchShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.batches = [list(range(i * batch_size, (i + 1) * batch_size)) for i in range(self.num_batches)]
        if len(dataset) % batch_size != 0:
            self.batches.append(list(range(self.num_batches * batch_size, len(dataset))))
        self.shuffle_batches()

    def shuffle_batches(self):
        self.shuffled_order = torch.randperm(len(self.batches)).tolist()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_indices = self.batches[self.shuffled_order[idx]]
        return torch.utils.data.Subset(self.dataset, batch_indices)

def collate_fn(batch):
    return torch.utils.data.default_collate([item for sublist in batch for item in sublist])
    
    

class Naive_Rehearsal(NormalNN):

    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 51200 #489728
        self.task_memory = {}
        self.skip_memory_concatenation = False

    def learn_batch(self, train_loader, val_loader,task_no):
  
        # 1.Combine training set
        if self.skip_memory_concatenation:
            new_train_loader = train_loader

        else: # default
            dataset_list = []
            for storage in self.task_memory.values():
                print("appending replay buffer of seen tasks of length: ",len(storage))
                dataset_list.append(storage)
            dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1
            dataset_list.append(train_loader.dataset)
            dataset = torch.utils.data.ConcatDataset(dataset_list)


            batch_shuffled_dataset = BatchShuffledDataset(dataset, batch_size=train_loader.batch_size)
            new_train_loader = torch.utils.data.DataLoader(batch_shuffled_dataset,
                                          batch_size=1,#train_loader.batch_size,
                                          shuffle=False,
                                          num_workers=train_loader.num_workers,
                                          collate_fn=collate_fn)
            

            
        # 2.Update model as normal
        super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader,task_no)
 
        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        
        # (c) Randomly choose some samples from new task and save them to the memory
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        self.task_memory[self.task_count] = Storage(train_loader.dataset, randind)
        # CHANGE THIS FOR CAUSALTTY ^^^^
        # TAKE DOUB;E IMAGES AND RANK THEM DOWN AND THEN TAKE X AMOUNT OF IMAGES


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):
    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = False
        self.n_fisher_sample = None
        self.empFI = False

    def learn_batch(self, train_loader, val_loader, task_no):
        # Perform Naive Rehearsal learning
        Naive_Rehearsal.learn_batch(self, train_loader, val_loader, task_no)
        
        # Perform EWC regularization
        if len(self.regularization_terms) > 0:
            self.update_regularization_terms(train_loader)
            
    def update_regularization_terms(self, train_loader):
        # Update the importance matrix for EWC
        importance = self.calculate_importance(train_loader)
        
        # Save the importance matrix and current parameters
        self.regularization_terms.append({'importance': importance, 'params': {n: p.clone().detach() for n, p in self.params.items()}})

    def calculate_importance(self, dataloader):
        # Calculate Fisher Information Matrix for EWC
        self.log('Computing EWC')
        print('Computing EWC')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms) > 0:
            importance = self.regularization_terms[-1]['importance']
        else:
            importance = {n: p.clone().detach().fill_(0) for n, p in self.params.items()}

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            self.log('Sample', self.n_fisher_sample, 'for estimating the F matrix.')
            print('Sample %s for estimating the F matrix.', self.n_fisher_sample)
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input.float())

            task_name = task[0]
            pred = preds[task_name][:, :self.valid_out_dim]
            ind = pred.max(1)[1].flatten()

            if self.empFI:
                ind = target

            loss = self.criterion(preds, target, task_name, regularization=False)

            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)

        return importance

    def criterion(self, preds, targets, tasks, regularization=True):
        # Custom criterion that combines EWC regularization with the original loss
        loss = super(Naive_Rehearsal, self).criterion(preds, targets, tasks)

        if regularization and len(self.regularization_terms) > 0:
            for reg_term in self.regularization_terms:
                importance = reg_term['importance']
                for n, p in self.params.items():
                    _loss = (importance[n] * (p - reg_term['params'][n]) ** 2).sum()
                    loss += self.agent_config['lambda'] * _loss

        return loss

        


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)


class GEM(Naive_Rehearsal):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        margin = self.config['reg_coef']
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def learn_batch(self, train_loader, val_loader,task_no):

        # Update model as normal
        super(GEM, self).learn_batch(train_loader, val_loader,task_no)

        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=128,#len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            
            
            # Only use the first batch
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if i == 0:  # Only take the first batch
                    if self.gpu:
                        mem_input = mem_input.cuda()
                        mem_target = mem_target.cuda()
                    self.task_mem_cache[t] = {'data': mem_input, 'target': mem_target, 'task': mem_task[0]}
                else:
                    break  # Exit after the first batch


            # assert len(mem_loader)==1,'The length of mem_loader should be 1'
            # for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
            #     if self.gpu:
            #         mem_input = mem_input.cuda()
            #         mem_target = mem_target.cuda()
            # self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

    def update_model(self, inputs, targets, tasks):

        # compute gradient on previous tasks
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), out
    

from torch.utils.data import Dataset

class CustomOutputDataset(Dataset):
    def __init__(self, dataset_pairs):
        self.dataset_pairs = dataset_pairs

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        outputs, labels = self.dataset_pairs[idx]
        return outputs, labels


class LGR(nn.Module):
    def __init__(self, agent_config, net, Gen, path, client_id):
        super(LGR, self).__init__()
        # self.Net = Net
        self.generator = Gen
        self.log = print if agent_config['print_freq'] > 0 else lambda *args: None  # Use a void function to replace the print
        self.config = agent_config
        if agent_config['gpuid'][0] > 0:
            self.gpu = True
            self.Device = torch.device("cuda")
        else:
            self.gpu = False
            self.Device = torch.device("cpu")
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        self.model = self.create_model(model=net)
        # self.generator=self,create_generator()
        self.criterion_fn = nn.MSELoss()
        # self.criterion_fn = nn.L1Loss()
        self.init_optimizer()
        if self.gpu:
            self.cuda()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.path = path
    
    # Set a interger here for the incremental class scenario
    def get_generator_weights(self):
        return self.generator.state_dict()
    
    def update_model(self, inputs, targets):
        # self.model=copy.deepcopy(model_)
        out = self.forward(inputs)
        loss = self.criterion(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out
    
    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'], gamma=0.1)
    
    def create_model(self, model, params=None):
        if self.gpu:
            model.cuda()
        if params is not None:
            params_dict = zip(model.conv_module.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.conv_module.load_state_dict(state_dict, strict=True)
        return model
    
    def forward(self, x):
        return self.model.forward(x)
    
    def set_generator_weights(self, weights):
        self.generator.load_state_dict(weights)
        
    def predict(net, trainloader, DEVICE, batch_size=16):
        new_pairs = []
        net.eval()  # Set the model to evaluation mode
        net.to(DEVICE)
        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = net(inputs)  # Forward pass through the model
                for i in range(len(outputs)):
                    new_pairs.append((outputs[i], labels[i]))
        # batch_size = 1
        new_data = CustomOutputDataset(new_pairs)
        new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
        return new_data_loader


    def predict_gen(self, net, trainloader, DEVICE, batch_size=16):
        new_pairs = []
        net.eval()  # Set the model to evaluation mode
        net.to(DEVICE)
        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = net(inputs)  # Forward pass through the model
                for i in range(len(outputs)):
                    new_pairs.append((outputs[i], outputs[i]))
        # batch_size = 1
        new_data = CustomOutputDataset(new_pairs)
        new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
        return new_data_loader
    
    def predict_from_gen(self, net, num_samples, DEVICE, batch_size=16):
        new_pairs = []
        net.eval()  # Set the model to evaluation mode
        net.to(DEVICE)
        with torch.no_grad():
            for i in range(num_samples):
                inputs = torch.randn(1, 64).to(DEVICE)
                outputs = net.decode(inputs)
                labels = self.model.fc_module(outputs)
                for i in range(len(outputs)):
                    new_pairs.append((outputs[i], labels[i]))
        # batch_size = 16
        new_data = CustomOutputDataset(new_pairs)
        new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
        return new_data_loader
    
    def predict_from_gen_gen(self, net, num_samples, DEVICE, batch_size=16):
        new_pairs = []
        net.eval()  # Set the model to evaluation mode
        net.to(DEVICE)
        with torch.no_grad():
            for i in range(num_samples):
                inputs = torch.randn(1, 64).to(DEVICE)
                outputs = net.decode(inputs)
                labels = self.model.fc_module(outputs)
                for i in range(len(outputs)):
                    new_pairs.append((outputs[i], outputs[i]))
        # batch_size = 16
        new_data = CustomOutputDataset(new_pairs)
        new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
        return new_data_loader
    
    def create_dataset(self, new_data):
        try:
            current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
        except:
            current_task_reconstucted_data = self.predict_from_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device, batch_size=16)
            torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
        # shuffle it with train_loader and return
        # both are dataloaders
        dataset1 = new_data.dataset
        dataset2 = current_task_reconstucted_data.dataset
        # for input, label in new_data:
        # 	print(input.shape)
        # 	print(label.shape)
        # 	break
        # for input, label in current_task_reconstucted_data:
        # 	print(input.shape)
        # 	print(label.shape)
        # 	break
        
        # Combine the datasets using ConcatDataset
        combined_dataset = ConcatDataset([dataset1, dataset2])
        
        # Create a DataLoader for the combined dataset with shuffling
        combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
        return combined_dataloader
    
    def create_dataset_gen(self, new_data):
        try:
            current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
        except:
            current_task_reconstucted_data = self.predict_from_gen_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device,
                                                                       batch_size=16)
            torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
        # shuffle it with train_loader and return
        # both are dataloaders
        dataset1 = new_data.dataset
        dataset2 = current_task_reconstucted_data.dataset
        # for input, label in new_data:
        # 	print(input.shape)
        # 	print(label.shape)
        # 	break
        # for input, label in current_task_reconstucted_data:
        # 	print(input.shape)
        # 	print(label.shape)
        # 	break
        
        # Combine the datasets using ConcatDataset
        combined_dataset = ConcatDataset([dataset1, dataset2])
        
        # Create a DataLoader for the combined dataset with shuffling
        combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
        return combined_dataloader
    
    def loss_function(self, recon_x, x, mu, logvar, input_dim):
        # MSE reconstruction loss
        # MSE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
        MSE = F.mse_loss(input=x.view(-1, input_dim), target=recon_x, reduction='mean')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    def train_generator(self, train_loader, task_count=0):
        print(" ............................................................................ Learning LGR Training Generator")
        
        # create dataset from latent features of self.model.root
        self.generator.train()
        self.generator.to(self.Device)
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        gen_train_data = self.predict_gen(self.model.conv_module, train_loader, self.Device, batch_size=16)
        if task_count != 0:
            gen_train_data = self.create_dataset_gen(gen_train_data)
        tot = self.config['schedule'][-1]
        for epoch in range(self.config['schedule'][-1] // 2):
            for input, target in gen_train_data:
                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()
                recon_batch, mu, logvar = self.generator.forward(input)
                loss = self.loss_function(recon_batch, input, mu, logvar, input.shape[1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # print loss
            print(f'Epoch [{epoch + 1}/{tot}], Loss: {loss.item():.4f}')
    
    def latent_creator(self, train_loader):
        self.model.eval()
        self.model.to(self.Device)
        current_task_latent_data = self.predict(self.model.conv_module, train_loader, self.Device, batch_size=16)
        print(" ............................................................................ Learning LGR Data Mixed")
        
        final = self.create_dataset(current_task_latent_data)
        return final
    
    def learn_batch(self, task_count, genweights, train_loader, learn_gen, val_loaderr=None):
        self.generator.load_state_dict(genweights)
        self.task_count = task_count
        self.log("Learning LGR")
        peak_ram = 0
        #ramu = RAMU()
        #peak_ram = max(peak_ram, ramu.compute("TRAINING"))
        if self.task_count == 0:
            print(" ............................................................................ Learning LGR Task 1")
            # Config the model and optimizer
            self.model.train()
            params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.001}]
            
            for epoch in range(self.config['schedule'][-1]):
                self.log('Epoch:{0}'.format(epoch))
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                self.scheduler.step(epoch)
                optimizer = torch.optim.Adam(params)
                
                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
                for i, (inputs, targets) in enumerate(train_loader):
                    data_time.update(data_timer.toc())
                    if self.gpu:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    out = self.forward(inputs)
                    loss = self.criterion(out, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #peak_ram = max(peak_ram, ramu.compute("TRAINING"))
                    
                    inputs = inputs.detach()
                    targets = targets.detach()
                    acc = 0
                    losses.update(loss, inputs.size(0))
                    batch_time.update(batch_timer.toc())
                    data_timer.toc()
                print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
        
        else:
            print(" ............................................................................ Learning LGR Task 2")
            
            # train(self.model.fc_module, train_loader, self.Device, self.config['schedule'][-1])
            mixed_task_data = self.latent_creator(train_loader)
            self.model.train()
            
            for epoch in range(self.config['schedule'][-1]):
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                
                # Config the model and optimizer
                self.log('Epoch:{0}'.format(epoch))
                self.scheduler.step(epoch)
                optimizer = torch.optim.Adam(self.model.fc_module.parameters(), lr=0.001)
                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
                for i, (inputs, targets) in enumerate(mixed_task_data):
                    data_time.update(data_timer.toc())
                    if self.gpu:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    # dont use self.update_model
                    # dont use self.forward
                    out = self.model.fc_module(inputs)
                    loss = self.criterion(out, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #peak_ram = max(peak_ram, ramu.compute("TRAINING"))
                    inputs = inputs.detach()
                    targets = targets.detach()
                    acc = 0
                    losses.update(loss, inputs.size(0))
                    batch_time.update(batch_timer.toc())
                    data_timer.toc()
                print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
            # freeze all layers of self.model.fc_module
            print(" ............................................................................ Learning LGR End to End Top Frozen")
            for epoch in range(self.config['schedule'][-1]):
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                
                # Config the model and optimizer
                self.log('Epoch:{0}'.format(epoch))
                self.model.train()
                # params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.001}]
                params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.0}]
                self.scheduler.step(epoch)
                optimizer = torch.optim.Adam(params)
                
                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
                for i, (inputs, targets) in enumerate(train_loader):
                    data_time.update(data_timer.toc())
                    if self.gpu:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    out = self.forward(inputs)
                    loss = self.criterion(out, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    inputs = inputs.detach()
                    targets = targets.detach()
                    acc = 0
                    losses.update(loss, inputs.size(0))
                    batch_time.update(batch_timer.toc())
                    data_timer.toc()
                print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
        
        if learn_gen == True:
            self.train_generator(train_loader, self.task_count)
        return peak_ram

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval
    