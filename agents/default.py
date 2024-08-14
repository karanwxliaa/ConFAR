from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, accuracy_au,accuracy_av, AverageMeter, Timer
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import mean_squared_error


class NormalNN(nn.Module):
	'''
	Normal Neural Network with SGD for classification
	'''
	def __init__(self, agent_config):
		'''
		:param agent_config (dict): task=str,lr=float,momentum=float,weight_decay=float,
									schedule=[int],  # The last number in the list is the end of epoch
									model_type=str,model_name=str,out_dim={task:dim},model_weights=str
									force_single_head=bool
									print_freq=int
									gpuid=[int]
		'''
		super(NormalNN, self).__init__()
		self.log = print if agent_config['print_freq'] > 0 else lambda \
			*args: None  # Use a void function to replace the print
		self.config = agent_config
		# If out_dim is a dict, there is a list of tasks. The model will have a head for each task.


		self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task

		if self.config['gpuid'][0] >= 0:
			print("using GPU with gpuid = ",self.config['gpuid'][0])
			self.gpu = True
			# self.cuda()
			self.device = torch.device("cuda")
		else:
			self.gpu = False
			self.device = torch.device("cpu")
		
		self.model = self.create_model() #check if we need to create or use existing model

		self.init_optimizer()
		
		self.reset_optimizer = False
		self.valid_out_dim = 0  # Default: 0

	def init_optimizer(self):
		optimizer_arg = {'params':self.model.parameters(),
						 'lr':self.config['lr'],
						 'weight_decay':self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD','RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'

		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
															  gamma=0.1)
	# In agents/default.py
	def create_model(self):

		#print("\n\n here ",self.config,"\n\n") 
		# cfg = self.config  
		# print("Available model types:", models.__dict__.keys())
		# print("Configuration model_type:", cfg['model_type'])
		# print("Configuration model_name:", cfg['model_name'])
		
		cfg = self.config
		cfg['model_type'] = cfg['model_type']
		# Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
		
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

		# Apply network surgery to the backbone
		# Create the heads for tasks (It can be single task or multi-task)
		n_feat = model.last.in_features
		#print("nfeat = ",n_feat)

		# The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
		# For a single-headed model the output will be {'All':output}
		model.last = nn.ModuleDict()
		for task,out_dim in cfg['out_dim'].items():
			
			model.last[task] = nn.Linear(n_feat,out_dim) #check for key and assign the output value
			#print("model.last for task ",task," is ",n_feat," & ",out_dim)

		# Redefine the task-dependent function
		def new_logits(self, x):
			outputs = {}
			for task, func in self.last.items():
				outputs[task] = func(x)
			return outputs

		# Replace the task-dependent function
		model.logits = MethodType(new_logits, model)
		
		# Load pre-trained weights
		if cfg['model_weights'] is not None:
			print('=> Load model weights:', cfg['model_weights'])
			model = model.load_weights(path=cfg['model_weights'])

			print('=> Load Done')
		
		return model.to(self.device)

	def forward(self, x):
		return self.model.forward(x)



	def predict(self, inputs):
		#cfg = self.config
		self.model.eval()
		
		out = self.forward(inputs.float())
		
		for t in out.keys():
			out[t] = out[t].detach()
		return out


	def validation_all(self, dataloader):
		batch_timer = Timer()
		acc_meter = AverageMeter()
		ccc_meter = AverageMeter()
		f1_meter = AverageMeter()

		batch_timer.tic()

		orig_mode = self.training
		self.eval()

		# Initialize containers for predictions and targets for each task
		av_outputs, av_targets = [], []
		fer_outputs, fer_targets = [], []
		au_outputs, au_targets = [], []
			

	   # Process data from the dataloader
		for i, (inputs, targets, task) in enumerate(dataloader):
			if self.gpu:
				with torch.no_grad():
					inputs = inputs.device()
					targets = targets.device()

			# Get predictions for the current batch
			output = self.predict(inputs)
			output = output[task[0]]

			# # Assuming 'All' task contains data for AU (first 12), FER (13th), and AV (last 2)
			# au_targets_batch = targets[:, :12]
			# fer_targets_batch = targets[:, 12:19]  # Single FER class
			# av_targets_batch = targets[:, 19:]

			# # Assuming the output structure matches: first part AU, then FER, then AV
			# au_outputs_batch = output[:, :12]
			# fer_outputs_batch = F.softmax(output[:, 12:19], dim=1)  # Assuming FER is output as logits
			# av_outputs_batch = output[:, 19:]

			# Assuming 'All' task contains data for AU (first 12), FER (13th), and AV (last 2)
			au_targets_batch = targets[:, :12]
			fer_targets_batch = targets[:, 12:13]  # Single FER class
			av_targets_batch = targets[:, 13:]

			# Assuming the output structure matches: first part AU, then FER, then AV
			au_outputs_batch = output[:, :12]
			fer_outputs_batch = output[:, 12:13]  # Assuming FER is output as logits
			av_outputs_batch = output[:, 13:]
			
			au_outputs_batch[au_outputs_batch >= 0.5] = 1
			au_outputs_batch[au_outputs_batch < 0.5] = 0   



			# Store results
			au_outputs_dict, fer_outputs_dict, av_outputs_dict = {},{},{}
			au_outputs_dict['All'] = au_outputs_batch         
			fer_outputs_dict['All'] = fer_targets_batch
			av_outputs_dict['All'] = av_targets_batch

			# Appending lists
			au_outputs.append(au_outputs_batch)
			au_targets.append(au_targets_batch)

			fer_outputs.append(torch.argmax(fer_outputs_batch, dim=1))  # Convert to predicted classes
			fer_targets.append(fer_targets_batch)

			av_outputs.append(av_outputs_batch)
			av_targets.append(av_targets_batch)
			

		# Concatenate all collected outputs and targets for each task
			
		# au_outputs = torch.cat(au_outputs, dim=0)
		# au_targets = torch.cat(au_targets, dim=0)
		# fer_outputs = torch.cat(fer_outputs, dim=0)
		# fer_targets = torch.cat(fer_targets, dim=0)
		# av_outputs = torch.cat(av_outputs, dim=0)
		# av_targets = torch.cat(av_targets, dim=0)

		self.train(orig_mode)  # Restore the original training mode

		# Call the specific validation functions for each task

		# av_metrics = self.validation_av((av_outputs, av_targets, 'AV'))
		# fer_metrics = self.validation_fer((fer_outputs, fer_targets, 'FER'))
		# au_metrics = self.validation_au((au_outputs, au_targets, 'AU'))

		ccc = accumulate_ccc(av_outputs_dict, av_targets_batch, task, ccc_meter)
		acc = accumulate_acc(fer_outputs_dict, fer_targets_batch, task, acc_meter)
		f1 = accumulate_acc_au(au_outputs_dict, au_targets_batch, task, f1_meter)

		logging.info(' * Val CCC {ccc.avg:.3f}, Val ACC  {acc.avg:.3f}, Val F1 {f1.avg:.2f}'
				.format(ccc=ccc_meter, acc=acc_meter, f1=f1_meter,time=batch_timer.toc()))
		self.log(' * Val CCC {ccc.avg:.3f}, Val ACC  {acc.avg:.3f}, Val F1 {f1.avg:.2f}'
				.format(ccc=ccc_meter, acc=acc_meter, f1=f1_meter,time=batch_timer.toc()))
 

		return acc.avg, f1.avg


	def validation_av(self, dataloader):
		batch_timer = Timer()
		ccc_meter = AverageMeter()
		f1_meter = AverageMeter()
		batch_timer.tic()

		orig_mode = self.training
		self.eval()
		output_list = []
		target_list = []

		for i, (input, target, task) in enumerate(dataloader):
			if self.gpu:
				with torch.no_grad():
					input = input.cuda()
					target = target.cuda()

			output = self.predict(input)
			predicted = output[task[0]]  # Assuming 'AV' is the key for this task
			output_list.append(predicted)
			
			target_list.append(target[:, :2])

			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			ccc = accumulate_ccc(output, target[:, :2], task, ccc_meter)
			
		outputs = torch.cat(output_list, dim=0)
		targets = torch.cat(target_list, dim=0)

		mse = mean_squared_error(
			torch.Tensor.cpu(targets).detach().numpy(),
			torch.Tensor.cpu(outputs).detach().numpy()
		)
		print("MSE: " + str(mse))

		self.train(orig_mode)

		logging.info(' * Val CCC {ccc.avg:.3f}, MSE Score {mse:.3f}, Total time {time:.2f}'
				.format(ccc=ccc_meter, mse=mse, time=batch_timer.toc()))
		self.log(' * Val CCC {ccc.avg:.3f}, MSE Score {mse:.3f}, Total time {time:.2f}'
				.format(ccc=ccc_meter, mse=mse, time=batch_timer.toc()))

		return ccc.avg, mse

		

	def validation_fer(self, dataloader):
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		batch_timer.tic()

		orig_mode = self.training
		self.eval()
		output_list = []
		target_list = []
		for i, (input, target, task) in enumerate(dataloader):

			if self.gpu:
				with torch.no_grad():
					input = input.cuda()
					target = target.cuda()
			
			output = self.predict(input)
			predicted = F.softmax(output[task[0]], dim=1)  #changed
			_, predicted = torch.max(predicted, 1)            
			output_list.append(predicted)
			target_list.append(target[:, 0].long())   
  
			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			acc = accumulate_acc(output, target[:, 0].long(), task, acc)
		outputs = torch.cat(output_list, dim=0)
		targets = torch.cat(target_list, dim=0) 
		f1 = f1_score(torch.Tensor.cpu(targets).detach().numpy(), torch.Tensor.cpu(outputs).detach().numpy(), average='weighted')
		self.train(orig_mode)

		logging.info(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
			  .format(acc=acc,time=batch_timer.toc()))
		self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
			  .format(acc=acc,time=batch_timer.toc()))
		return acc.avg, f1
	

	
	
	def validation_au(self, dataloader):
		import numpy as np
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		batch_timer.tic()

		orig_mode = self.training
		self.eval()
		output_list = []
		target_list = []

		for i, (input, target, task) in enumerate(dataloader):
  

			if self.gpu:
				with torch.no_grad():
					input = input.cuda()
					target = target.cuda()

			output = self.predict(input)

			outputs = output["AU"]
			outputs[outputs >= 0.5] = 1
			outputs[outputs < 0.5] = 0  
			output["AU"] = outputs

			output_list.append(outputs)
			target_list.append(target)   
  
			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			acc = accumulate_acc_au(output, target, task, acc)

		outputs = torch.cat(output_list, dim=0)
		targets = torch.cat(target_list, dim=0) 
		N_val,C_val = targets.shape
		
		f1_val = np.zeros((C_val,1))
		for kk in range(C_val):    
			f1_val[kk] = f1_score(targets[kk].cpu().detach().numpy(), outputs[kk].cpu().detach().numpy(), average="binary")
		#print("f1 score: " + str(f1_val.mean()))
		self.train(orig_mode)

		logging.info(' * Val f1 {acc.avg:.3f}, Total time {time:.2f}'
			  .format(acc=acc,time=batch_timer.toc()))
		self.log(' * Val f1 {acc.avg:.3f}, Total time {time:.2f}'
			  .format(acc=acc,time=batch_timer.toc()))
		return acc.avg, f1_val.mean()
	
	def validation(self, dataloader,val_task):
		if val_task=="FER":
			return self.validation_fer(dataloader)
		if val_task=="AV":
			return self.validation_av(dataloader)
		if val_task=="AU":
			return self.validation_au(dataloader)
		if val_task=='All':
			return self.validation_all(dataloader)
		

	def weighted_multi_label_loss(self,preds, targets):
		class_weights = torch.tensor([1.219, 1.506, 1.271, 0.559, 0.469, 0.433, 0.458, 0.553, 1.521, 0.75, 1.557, 1.702]).to(self.device)
		
		lambda_c = torch.tensor(0.5).to(self.device)
		m_pos = torch.tensor(0.9).to(self.device)
		m_neg = torch.tensor(0.1).to(self.device)
		const = torch.zeros(12).to(self.device)
		
		preds = preds.to(self.device)
		targets = targets.to(self.device)
		# targets = targets.cpu().detach().numpy()
		# preds = preds.cpu().detach().numpy()

		#print("TARGERTS =", targets.shape,"\n\nPREDS= ",preds.shape)
		#print("TARGERTS =", targets,"\n\nPREDS= ",preds)

		L = class_weights * (targets * torch.square(torch.max(const, m_pos - preds)) + lambda_c * (torch.tensor(1.0).to(self.device) - targets) * torch.square(
			torch.max(const, preds - m_neg)))
		
		
		return torch.mean(L)
	

	def criterion(self, preds, targets, task, **kwargs):
		# loss = 0
		
		t_target = targets
		t_preds = preds[task]

		if task == 'FER':
			t_target = targets[:, 0].long()
			loss = self.criterion_fn(t_preds, t_target)

		elif task == 'AV':
			t_target = targets[:, :2]	
			loss = self.criterion_fn(t_preds, t_target)	

		elif task == 'AU':
			t_target = targets
			loss = self.weighted_multi_label_loss(t_preds, t_target)

		
		return loss

	


	def update_model(self, inputs, targets, task):

		out = self.forward(inputs)
		
		
		loss = self.criterion(out, targets, task)
		
		#loss = self.criterion(out, targets, task)
		#print("loss in update model = ",loss)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out
	


	def learn_batch(self, train_loader, val_loader, current_task):

		if current_task=="FER":
			self.criterion_fn = nn.CrossEntropyLoss()
		elif current_task=="AU":
			#class_weights = [1.219, 1.506, 1.271, 0.559, 0.469, 0.433, 0.458, 0.553, 1.521, 0.75, 1.557, 1.702]
			self.criterion_fn = nn.MultiLabelMarginLoss()
		elif current_task=="AV":
			self.criterion_fn = torch.nn.MSELoss()
		elif current_task=="All":
			self.criterion_fn1 = torch.nn.MSELoss()
			self.criterion_fn2 = nn.CrossEntropyLoss()
		print("\n\nsent model to ",self.device)
		self.model = self.model.to(self.device)
		
		if self.reset_optimizer:  # Reset optimizer before learning each task

			
			self.log('Optimizer is reset!')
			self.init_optimizer()
			
		flag = 0
		for epoch in range(self.config['schedule'][-1]):
			data_timer = Timer()
			batch_timer = Timer()
			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()
			acc = AverageMeter()
			ccc = AverageMeter()
			f1 = AverageMeter()

			
			if (flag==1):
				break
			
			logging.info('Epoch:{0}'.format(epoch))
			# Config the model and optimizer
			self.log('Epoch:{0}'.format(epoch))
			self.model.train()
			self.scheduler.step(epoch)
			for param_group in self.optimizer.param_groups:
				self.log('LR:',param_group['lr'])

			# Learning with mini-batch
			data_timer.tic()
			batch_timer.tic()

			# if current_task=="FER":
			# 	logging.info('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
			# 	self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
		
			# elif current_task=="AU":
			# 	logging.info('Itr\t\tTime\t\t  Data\t\t  Loss\t\tF1')
			# 	self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tF1')

				
			# elif current_task=="AV":
			# 	logging.info('Itr\t\tTime\t\t  Data\t\t  Loss\t\tCCC')
			# 	self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tCCC')

			# elif current_task=="All":
			logging.info('Itr\t\t Time\t\t Data\t\t Loss\t\t CCC\t\t ACC\t\t F1')
			self.log('Itr\t\t Time\t\t Data\t\t Loss\t\t CCC\t\t ACC\t\t F1')
		
			for i, (input, target, task_name) in enumerate(train_loader):
				
				#print("Task = ",task_name)
				# print("Input = ",input)
				# print("Target = ",target)

				input = input.to(self.device)
				target = target.to(self.device)
				
				data_time.update(data_timer.toc())  # measure data loading time
				
				loss, output = self.update_model(input, target, task_name[0])

				
				input = input.detach()
				target = target.detach()

			 
				if task_name[0] == 'FER':
					#print("target from loader = ",target)
					acc = accumulate_acc(output, target[:, 0].long(), task_name, acc)

				elif task_name[0] == 'AV':
					ccc = accumulate_ccc(output, target[:, :2], task_name, ccc)


				elif task_name[0] == 'AU':
					# measure accuracy and record loss
					outputs = output["AU"]
					outputs[outputs >= 0.5] = 1
					outputs[outputs < 0.5] = 0   
					output["AU"] = outputs  
					f1 = accumulate_acc_au(output, target, task_name, f1)

				
				losses.update(loss, input.size(0))

				batch_time.update(batch_timer.toc())  # measure elapsed time
				data_timer.toc()

				

				if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
					
					if current_task == 'FER':
						logging.info('[{0}/{1}]\t'
						  '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '"NA" ("NA")\t'
						  '{acc.val:.2f} ({acc.avg:.2f})\t'
						  '"NA" ("NA")'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,acc=acc))
						self.log('[{0}/{1}]\t'
						  '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '"NA" ("NA")\t'
						  '{acc.val:.2f} ({acc.avg:.2f})\t'
						  '"NA" ("NA")'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,acc=acc))

					elif current_task == 'AV':
						logging.info('[{0}/{1}]\t'
						  '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '{ccc.val:.2f} ({ccc.avg:.2f})\t'
						  '"NA" ("NA")\t'
						  '"NA" ("NA")'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,ccc=ccc))
						self.log('[{0}/{1}]\t'
							'{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '{ccc.val:.2f} ({ccc.avg:.2f})\t'
						  '"NA" ("NA")\t'
						  '"NA" ("NA")'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,ccc=ccc))

					elif current_task == 'AU':
						logging.info('[{0}/{1}]\t'
						  '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '"NA" ("NA")\t'
						  '"NA" ("NA")\t'
						  '{f1.val:.2f} ({f1.avg:.2f})'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,f1=f1))
						self.log('[{0}/{1}]\t'
							'{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
						  '{loss.val:.3f} ({loss.avg:.3f})\t'
						  '"NA" ("NA")\t'
						  '"NA" ("NA")\t'
						  '{f1.val:.2f} ({f1.avg:.2f})'.format(
						i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses,f1=f1))
											
			if current_task == 'FER':
				self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
				logging.info(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

			elif current_task == 'AV':
				self.log(' * Train CCC {ccc.avg:.3f}'.format(ccc=ccc))
				logging.info(' * Train CCC {ccc.avg:.3f}'.format(ccc=ccc))

			elif current_task == 'AU':
				self.log(' * Train F1 {f1.avg:.3f}'.format(f1=f1))
				logging.info(' * Train F1 {f1.avg:.3f}'.format(f1=f1))

			elif current_task == 'All':
				self.log(' * Train CCC {ccc.avg:.3f}'.format(ccc=ccc))
				logging.info(' * Train CCC {ccc.avg:.3f}'.format(ccc=ccc))
				self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
				logging.info(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
				self.log(' * Train F1 {f1.avg:.3f}'.format(f1=f1))
				logging.info(' * Train F1 {f1.avg:.3f}'.format(f1=f1))

			# UPADTE THIS FOR FER AND AU
			if val_loader != None:
				if current_task == 'FER':
					axx, _ = self.validation_fer(val_loader)
				elif current_task == 'AV':
					axx, _ = self.validation_av(val_loader)
				elif current_task == 'AU':
					axx, _ = self.validation_au(val_loader)
				elif current_task == 'All':
					t1, t2 = self.validation_all(val_loader)
					axx, _ = t2,t1
				
	
	def learn_stream(self, data, label):
		assert False,'No implementation yet'

	def add_valid_output_dim(self, dim=0):
		# This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
		self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
		logging.info('Incremental class: Old valid output dimension: %s', self.valid_out_dim)
		if self.valid_out_dim == 'ALL':
			self.valid_out_dim = 0  # Initialize it with zero
		self.valid_out_dim += dim
		self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
		logging.info('Incremental class: New Valid output dimension: %s', self.valid_out_dim)
		return self.valid_out_dim

	def count_parameter(self):
		return sum(p.numel() for p in self.model.parameters())

	def save_model(self, filename):
		model_state = self.model.state_dict()
		if isinstance(self.model,torch.nn.DataParallel):
			# Get rid of 'module' before the name of states
			model_state = self.model.module.state_dict()
		for key in model_state.keys():  # Always save it to cpu
			model_state[key] = model_state[key].cpu()
		print('=> Saving model to:', filename)
		torch.save(model_state, filename + '.pth')
		print('=> Save Done')

	def cuda(self):
		torch.cuda.set_device(self.config['gpuid'][0])
		self.model = self.model.cuda()
		self.criterion_fn = self.criterion_fn.cuda()
		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
		return self

def accumulate_acc(output, target, task, meter):

	for t, t_out in output.items():
		inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
		if len(inds) > 0:
			t_out = t_out[inds]
			t_target = target[inds]
			meter.update(accuracy(t_out, t_target), len(inds))

	return meter




def accumulate_ccc(output, target, task, meter):
	
	for t, t_out in output.items():
		inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched a specific task
		if len(inds) > 0:
			t_out = t_out[inds]
			t_target = target[inds]
			# print(len(t_out))
			# print(len(t_target))
			meter.update(accuracy_av(t_out,  t_target), len(inds))

			#meter.update(accuracy_av(t_out, t_target,av="a"), len(inds))
			#meter.update(accuracy_av(t_out, t_target,av="v"), len(inds))

	return meter


def accumulate_acc_au(output, target, task, meter):

	for t, t_out in output.items():
		inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
		if len(inds) > 0:
			t_out = t_out[inds]
			t_target = target[inds]
			meter.update(accuracy_au(t_out, t_target), len(inds))
	return meter

