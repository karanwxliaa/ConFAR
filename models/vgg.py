# import torch
# import torch.nn as nn
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchfile
# import torchvision.models as models

# class VGG_16(nn.Module):
#     """
#     Main Class
#     """

#     def __init__(self,num_features=24):
#         """
#         Constructor
#         """
#         super().__init__()
#         self.block_size = [2, 2, 3, 3, 3]
#         self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
#         self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
#         self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
#         self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
#         self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
#         self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.fc6 = nn.Linear(512 * 7 * 7, 4096)
#         self.fc7 = nn.Linear(4096, 4096)
#         self.fc8 = nn.Linear(4096, 256)
#         self.last = nn.Linear(256, num_features)


#     def load_weights(self, path="pretrained/VGG_FACE.t7"):
#         """ Function to load luatorch pretrained

#         Args:
#             path: path for the luatorch pretrained
#         """
#         model = torchfile.load(path)
#         counter = 1
#         block = 1
#         for i, layer in enumerate(model.modules):
#             if layer.weight is not None:
#                 if block <= 5:
#                     self_layer = getattr(self, "conv_%d_%d" % (block, counter))
#                     counter += 1
#                     if counter > self.block_size[block - 1]:
#                         counter = 1
#                         block += 1
#                     self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
#                     self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
#                 else:
#                     self_layer = getattr(self, "fc%d" % (block))
#                     block += 1
#                     self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
#                     self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]


#     def forward(self, x):
#         """ Pytorch forward

#         Args:
#             x: input image (224x224)

#         Returns: class logits

#         """
#         x = F.relu(self.conv_1_1(x))
#         x = F.relu(self.conv_1_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2_1(x))
#         x = F.relu(self.conv_2_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_3_1(x))
#         x = F.relu(self.conv_3_2(x))
#         x = F.relu(self.conv_3_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_4_1(x))
#         x = F.relu(self.conv_4_2(x))
#         x = F.relu(self.conv_4_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_5_1(x))
#         x = F.relu(self.conv_5_2(x))
#         x = F.relu(self.conv_5_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc6(x))
#         x = F.dropout(x, 0.5, self.training)
#         x = F.relu(self.fc7(x))
#         x = F.dropout(x, 0.5, self.training)


#         out = {}
#         # Forward pass through the modified VGG16 model
#         x = self.vgg16_face(x)

#         try:
#             out['FER'] = self.last['FER'](x)
#         except:
#             pass 

#         try:
#             out['AV'] = self.last['AV'](x)
#         except:
#             pass 

#         try:
#             out['AU'] = self.last['AU'](x)
#         except:
#             pass 

#         return out

    

# class VGG16(nn.Module):
#     def __init__(self, num_features=24, num_classes_fer=7, num_classes_au=12):
#         super(VGG16, self).__init__()
#         # Load the pre-trained VGG16 model
#         self.vgg16_face = models.vgg16(pretrained=True)
#         # Modify the classifier of the pre-trained model
#         # Assuming the original VGG16 has 4096 units in its penultimate layer
#         num_features_before_classifier = 4096

#         # Replace the classifier of the VGG16 model
#         # Remove the last layer (classifier) and add your own layers
#         self.vgg16_face.classifier = nn.Sequential(
#             *list(self.vgg16_face.classifier.children())[:-1],  # Keep all layers except the last one
#             nn.Linear(num_features_before_classifier, 256)  # Your custom layer
#         )

#         self.last = nn.Linear(256, num_features)
#     def forward(self, x):

        
#         out = {}

#         # Forward pass through the modified VGG16 model
#         x = self.vgg16_face(x)

#         try:
#             out['FER'] = self.last['FER'](x)
#         except:
#             pass 

#         try:
#             out['AV'] = self.last['AV'](x)
#         except:
#             pass 

#         try:
#             out['AU'] = self.last['AU'](x)
#         except:
#             pass 

#         return out


# if __name__ == "__main__":
#     model = VGG_16().double()
#     model.load_weights()



import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_features=24, num_classes_fer=7, num_classes_au=12):
        super(VGG16, self).__init__()
        # Load the pre-trained VGG16 model
        self.vgg16_face = models.vgg16(pretrained=True)
        # Modify the classifier of the pre-trained model
        # Assuming the original VGG16 has 4096 units in its penultimate layer
        num_features_before_classifier = 4096

        # Replace the classifier of the VGG16 model
        # Remove the last layer (classifier) and add your own layers
        self.vgg16_face.classifier = nn.Sequential(
            *list(self.vgg16_face.classifier.children())[:-1],  # Keep all layers except the last one
            nn.Linear(num_features_before_classifier, 256)  # Your custom layer
        )

        self.last = nn.Linear(256, num_features)

        # can also freeze the convolutional layers to retain pre-trained features
        # for param in self.vgg16_face.features.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        
        out = {}

        # Forward pass through the modified VGG16 model
        x = self.vgg16_face(x)

        try:
            out['FER'] = self.last['FER'](x)
        except:
            pass 

        try:
            out['AV'] = self.last['AV'](x)
        except:
            pass 

        try:
            out['AU'] = self.last['AU'](x)
        except:
            pass 

        return out

def custom_vgg16_face():
    return VGG16()

