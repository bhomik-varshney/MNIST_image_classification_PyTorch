import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
t1 = torch.tensor(4.)
print(t1)
print(t1.dtype)
t2 = torch.tensor([1,2,4.0])   # it wants every data type of an element to be same in this case it is floating value.
t3 = torch.tensor([[1,2,3],[3,4,5],[6,7,8]])  # 2-d tensor
print(t3.shape)
t4 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[0,4,5],[6,4,6],[3,0,1]]])  # 3-d tensor.
print(t4.shape)

print(t1*t2)  #simple arithematic operations of matrices.

w = torch.tensor([4.0],requires_grad=True)
x = torch.tensor([5.0])
b = torch.tensor([6.],requires_grad=True)
y = w*x + b
y.backward() #for the derivative computation
print(y)
print('dy/dx', x.grad)  #for x it is none as we did not set reuires_grad = True
print('dy/dw',w.grad)
print('dy/db',b.grad)


n1 = np.array([1,2,3,4])
t5 = torch.from_numpy(n1)
print(t5) #converting the numpy array into pytorch tensor.
print(n1.dtype,t5.dtype)
n2 = t5.numpy()  #converting back into numpy format
w.grad.zero_()  #converting grad to zero after completion and moving ahead.
print(w.grad)


#dumb linear regression model for analysis :-
x_train = torch.tensor([[2.,3,4],[4,5,6]])
x_targets = torch.tensor([[3,4.],[4,5]])
w1 = torch.randn((3,2),requires_grad=True)  #make a random tensor of size (2,3)
b1 = torch.randn((2,2),requires_grad=True)
def model(x):
    return x @ w1 + b1
preds = model(x_train)
print(preds)
print(x_targets)
def mse(l1,l2):
    l3 = l1-l2
    return torch.sum(l3*l3)/l3.numel()

loss= mse(x_targets,preds)
print(loss)   #it has very high mse value. i.e around 10.
loss.backward()
print(w1.grad, b1.grad)

#inbuilt-linear regression function:-

import torch.nn as nn

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
from torch.utils.data import TensorDataset
train_ds = TensorDataset(inputs,targets)
print(train_ds[0:3])   #it will show 3 entries form inputs and 3 entries from targets relative to inputs.

from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds,batch_size=5,shuffle=True)

model = nn.Linear(3,2) #we have input features as temperature,rainfall,humidity and output features as apples and oranges.
print(model.weight)
print(model.bias)

preds = model(inputs)
print(preds)

#calculating loss

import torch.nn.functional as F
loss = F.mse_loss(preds,targets)
print(loss)
opt = torch.optim.SGD(model.parameters(),lr = 1e-5)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
fit(100,model,F.mse_loss,opt,train_dl)
pred = model(inputs)
print(pred)
print(targets)
loss1 = F.mse_loss(pred,targets)
print(loss1)   #using optimization loss reudced to 40.531



# image processing :-

import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.transforms import transforms  #converting images into tensors.
from torch.utils.data import random_split
# It consists of 28px by 28px grayscale images of handwritten digits (0 to 9), along with labels for each image indicating which digit it represents.

dataset = MNIST(root= './data/mnist',download = True)
print(dataset)
test_dataset = MNIST(root = './data/mnist',train = False)
print(test_dataset[0])
print(dataset[0])

# index = int(input('enter the index of the image : '))
# image,label = dataset[index]
# plt1 = plt.imshow(image, cmap = 'gray')
# plt.title(f"Label : {label}")
# plt.show()


dataset = MNIST(root='./data/mnist',train=True,transform = transforms.ToTensor() )
index1 = int(input('enter the index of the image : '))
imgTensor, label = dataset[index1]
print(imgTensor.shape,label)   #torch.Size([1, 28, 28]) 28x28 pixels.

print(imgTensor[:,10:15,10:15])  # a small portion of the image into tensor.
# 0 denotes absolute black and 1 denotes absolute white color in image. 0 to 1 represents different shades pf gray.
# torch.max and torch.min is used to know the maximum and minimum value of an element present in a tensor respectively.
# plt.imshow(imgTensor[0,10:15,10:15],cmap ='gray')
# plt.show()
# splitting our data into two parts training and validation datasets.

train_df , val_df= random_split(dataset,[50000,10000])
print(len(train_df),len(val_df))
print(len(dataset))

train_loader = DataLoader(train_df,shuffle = True,batch_size= 128)
val_loader = DataLoader(val_df,batch_size = 128)

# logistic Regression Model :-
model1 = nn.Linear(28*28,10)
print(model1.weight.shape)
print(model1.bias.shape)
for images,labels in train_loader:
    print(labels)
    print(images.shape)
    break


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

model4 = MnistModel()
for images,labels in train_loader:
    print('images.shape',images.shape)
    outputs = model4(images)
    break
print('outputs.shape',outputs.shape)
print('sample outputs',outputs[:2].data)
exps = torch.exp(outputs[0])   # but they are not probabilities cause they also have negative values so to convert these biases into probabilities we will make it to soft max function.
prob = exps / torch.sum(exps)  # that's a softmax function.
print(prob)

import torch.nn.functional as F
probs = F.softmax(outputs,dim= 1)  # inbult softmax function
print(probs[:2])

max_probs, preds = torch.max(probs,dim =1)
print(preds)
print(max_probs)
print(labels)
def accuracy(outputs,labels):
    max_prob,preds = torch.max(outputs,dim = 1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
accuracy1 = accuracy(outputs,labels)
print(accuracy1)
# a commonly used loss function for classification problems is cross entropy.
# cross entropy function also first performs softmax function.
loss_fn = F.cross_entropy(outputs,labels)
print(loss_fn)

#model training:
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model = MnistModel()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
print(evaluate(model,val_loader))
history = fit(5,0.001,model,train_loader,val_loader)   # we can see that the val accuracy is increasing as we are trainignour model.
print(history)
accuracies = [r['val_acc'] for r in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');
plt.show()   # we can see that our model is saturating to its limit.

# testing with individual images.
test_dataset = MNIST(root = './data/mnist',train = False, transform = transforms.ToTensor())
def predict_image(img,model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _,pred = torch.max(yb,dim = 1)
    return pred[0].item()

index2 = int(input('input the value of index of the image : '))
img,label = dataset[index2]
plt.imshow(img[0],cmap = 'gray')
plt.title(label)
print('label :',label, 'predicted_label : ', predict_image(img,model))
plt.show()

# At last, we hit the maximum val_accuracy(or test accuracy) of 79.08 %.


# now using feed forward neural network:
# there we will use a secondary logistic regression layer and an intermediate layer of non-linear function (relu)

class MnistModel1(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)


    def forward(self, xb):
        xb = xb.view(xb.size(0),-1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model5 = MnistModel1(784,32,10)
history = [evaluate(model5,val_loader)]
print(history)
history += fit(5,0.1,model5,train_loader,val_loader) # we got 93.55 percent val accuracy maximum with lr = 0.1.
losses = [x['val_loss'] for x in history]
plt.plot(losses,'-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss vs No. of epochs')
plt.show()

for t in model5.parameters():
    print(t.shape)
for images, labels in train_loader:
    outputs = model5(images)
    loss = F.cross_entropy(outputs,labels)
    print('loss',loss.item())
    break
print('output shape : ',outputs.shape)
print('sample outputs : ' , outputs[0:2].data)

print(torch.cuda.is_available())

# we can surely improve our model by increasing hidden size, adding more hidden layers, etc.
# (sigmoid function)

# we can also use nn.sequential(we can give list of layers to it) and can use convolutional neural networks to increase our accuracy of the model.