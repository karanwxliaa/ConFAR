from torchvision import transforms


def load_transforms(image_size=224, aug=False):
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	
	else:
		train_transform = transforms.Compose([
			transforms.Resize((image_size, image_size)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	return train_transform, test_transform