import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

config = {'lr':1e-3,
          'wd':1e-2,
          'bs':256,
          'img_size':512,
          'epochs':100,
          'seed':1000}

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=config['seed'])

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(config['img_size'],config['img_size'],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ])

class ImageNetDataset(Dataset):
    def __init__(self,paths,augmentations):
        self.paths = paths
        self.augmentations = augmentations
    
    def __getitem__(self,idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image
    
    def __len__(self):
        return len(self.paths)

class VQ(nn.Module):
    
    def __init__(self,num_embeddings=512,embedding_dim=64,commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings,self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
    
    def forward(self,inputs):
        inputs = inputs.permute(0,2,3,1).contiguous()
        input_shape = inputs.shape
        
        flat_inputs = inputs.view(-1,self.embedding_dim)
        
        distances = torch.cdist(flat_inputs,self.embeddings.weight)
        encoding_index = torch.argmin(distances,dim=1) 
        
        quantized = torch.index_select(self.embeddings.weight,0,encoding_index).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        c_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        quantized = quantized.permute(0,3,1,2).contiguous()
        return c_loss, quantized

class ResudialBlock(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels):
        super(ResudialBlock,self).__init__()
        self.resblock = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(hidden_channels,out_channels,kernel_size=1,stride=1,bias=False))
    def forward(self,x):
        return x + self.resblock(x)

class ResudialStack(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels,num_res_layers):
        super(ResudialStack,self).__init__()
        self.num_res_layers = num_res_layers
        self.layers = nn.ModuleList([ResudialBlock(in_channels,out_channels,hidden_channels) for _ in range(num_res_layers)])
    
    def forward(self,x):
        for i in range(self.num_res_layers):
            x = self.layers[i](x)
        return F.relu(x)

class Model(nn.Module):

    def __init__(self,num_embeddings=512,embedding_dim=64,commitment_cost=0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        #encode
        self.conv1 = nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.resblock1 = ResudialStack(128,128,64,3)
        
        #vq 
        self.vq_conv = nn.Conv2d(128,self.embedding_dim,kernel_size=1,stride=1)
        self.vq = VQ(self.num_embeddings,self.embedding_dim,self.commitment_cost)
        
        #decode
        self.conv4 = nn.Conv2d(self.embedding_dim,64,kernel_size=3,stride=1,padding=1)
        self.resblock2 = ResudialStack(64,64,32,3)
        self.conv5 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.conv6 = nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1)


    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.resblock1(x)
        return x
        
    def decode(self,quantized):
        x = self.conv4(quantized)
        x = self.resblock2(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def forward(self,inputs):
        x = self.encode(inputs)
        c_loss,quantized =  self.vq(self.vq_conv(x))
        outputs = self.decode(quantized)
        rec_loss = F.mse_loss(outputs,inputs)
        loss = rec_loss + c_loss
        return loss,outputs,rec_loss

model = Model()
model.load_state_dict(torch.load('model/model2.bin',map_location='cpu'))
    
def get_rec_image(filename):
    test_path = f'static/images/{filename}'
    
    image = cv2.imread(test_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmentations = get_train_transforms()
    augmented = augmentations(image=image)
    image = augmented['image']
    image = image.unsqueeze(0)
    _,predictions,_ = model(image)
    predictions *= 255
    predictions = predictions.squeeze(0)
    rec_image = predictions.detach().cpu().permute(1,2,0).numpy()

    rec_image = cv2.cvtColor(rec_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('static/images/image.jpg',rec_image)
