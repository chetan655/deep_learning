import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
print(f"pytorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # we we want train dataset or test dataset
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None  # how do we want to transform the labels/targets
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

len(train_data), len(test_data)

# see the first trining examples
class_names = train_data.classes
class_names

# index
class_to_idx = train_data.class_to_idx
class_to_idx

# check the shape of image
image, label = train_data[0]
image.shape # this will print color_channels, height, width
label

image, label = train_data[0]
print(f"Imge shape: {image.shape}")
plt.imshow(image.squeeze())   # we have to use squeeze() to remove one extra dimension that is for color channel
plt.title(label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])


# plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
  random_idx = torch.randint(0, len(train_data), size=[1]).item()  # this will give one random index and convert into python int
  img, label = train_data[random_idx]
  fig.add_subplot(rows, cols, i)
  plt.imshow(img.squeeze(), cmap="gray")
  plt.title(class_names[label])
  plt.axis(False)


from torch.utils.data import DataLoader

# setup the batch size hyperparameter
BATCH_SIZE = 32

# turn datasets into interables
train_dataLoader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataLoader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_dataLoader, test_dataLoader


# lets check what we have created
print(f"DataLoaders: {train_dataLoader, test_dataLoader}")
print(f"Length of train_dataLoader: {len(train_dataLoader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataLoader: {len(test_dataLoader)} batches of {BATCH_SIZE}")



# check inside of training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataLoader))
train_features_batch.shape, train_labels_batch.shape


# create a flatten layer
flatten_model = nn.Flatten()   # flatten turns multi-dimensional data into single dimension

# get a single sample
x = train_features_batch[0]

# flatten the sample
output = flatten_model(x)

# print
print(f"shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"shape after flattening: {output.shape} -> [color_channels, features(height*width)]")



class FashionMNISTModelV0(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape,
                  out_features=hidden_units),
        nn.Linear(in_features=hidden_units,
                  out_features=output_shape)
    )

  def forward(self, x):
    return self.layer_stack(x)
  




# creating instance
torch.manual_seed(42)

model_0 = FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
).to("cpu")

model_0


dummy_x = torch.rand([1,1,28,28])
model_0(dummy_x)



import requests
from pathlib import Path

# Download helper functin
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
  print("Download successful.")




# import accuracy metric from helper_functions.py
from helper_functions import accuracy_fn

# setup loss function
loss_fn = nn.CrossEntropyLoss()

# setup optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)



# creating a function to time our experiments

from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
# print difference between start and end time
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time



start_time = timer()

end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")




# import tqdm for progress bar
from tqdm.auto import tqdm

# set the manual seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# set the number of epochs
epochs = 3

# create training and test loop
for epoch in tqdm(range(epochs)):
  print(f"epoch: {epoch}\n---")
  ### Training
  train_loss = 0
  # add a loop to loop through training batches
  for batch, (X, y) in enumerate(train_dataLoader):
    model_0.train()
    # forward pass
    y_pred = model_0(X)

    # calculate loss (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss 

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

    # print
    if batch % 400 == 0:
      print(f"looked at {batch * len(X)}/{len(train_dataLoader.dataset)} samples.")

  # divide total train lsos by length of train dataloader
  train_loss /= len(train_dataLoader)

  ### testing
  test_loss, test_acc = 0, 0
  model_0.eval()
  with torch.inference_mode():
    for X_test, y_test in test_dataLoader:
      # forward pass
      test_pred = model_0(X_test)

      # calculate loss
      test_loss += loss_fn(test_pred, y_test)

      # calculate acc
      test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

    # calculate  the test loss average per batch
    test_loss /= len(test_dataLoader)

    # calculate test acc average per batch
    test_acc /= len(test_dataLoader)
  
  # print
  print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

# calculate training time
train_time_end_on_cpu = timer()

total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=str(next(model_0.parameters()).device))
 


str(next(model_0.parameters()).device)






# make prediction and get model 0 results
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
  # return a dict containing the results of model predicting on data loader.
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      # make predictions
      y_pred = model(X)

      # accumulate the loss and accuracy values per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    # scale the loss and acc to find the avg loss and acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__,  # only works when model was created with a class 
          "model_loss": loss.item(),
          "model_acc": acc}

# calculate model_0 results on test dataset
# model_0_results = eval_model(model=model_0,
#                              data_loader=test_dataLoader,
#                              loss_fn=loss_fn,
#                              accuracy_fn=accuracy_fn)
# model_0_results




# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device




# creating a model with nonn-linear and linear layers
class FashionMNISTModelV1(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape,
                  out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units,
                  out_features=output_shape),
        nn.ReLU()
    )

  def forward(self, x:torch.Tensor):
    return self.layer_stack(x)
  




# creating an instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
next(model_1.parameters()).device




# loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)




def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
  """Performs a training with model trying to learn on data_loader."""
  train_loss, train_acc = 0, 0

  # put model into training mode
  model.train()

  for batch, (X, y) in enumerate(data_loader):
    # put data on target device
    X, y = X.to(device), y.to(device)

    # forward pass (this will return logits but out loss and accuracy function expects out prediction in same shape as true values)
    y_pred = model(X)

    # calculate loss and accuracy (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step -> updata the model's parameters once *per batch*
    optimizer.step()

  # divide total train loss and acc by length of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")





def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
  """performs a testing loop step on model going over data_loader"""
  test_loss, test_acc = 0, 0
  # put model in eval mode
  model.eval()

  # turn on inference_mode context manager
  with torch.inference_mode():
    for X, y in data_loader:
      # send data to the target device
      X, y = X.to(device), y.to(device)

      # forward pass
      test_pred = model(X)

      # calculate the loss/acc
      loss = loss_fn(test_pred, y)
      test_loss += loss
      test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

    # Adjust matrix and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    # print
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")




torch.manual_seed(42)

# measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

# set epochs
epochs = 3

# create a optimization and evaluation loop using train_step() and test_step()
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------------")
  train_step(model=model_1,
             data_loader=train_dataLoader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model_1,
            data_loader=test_dataLoader,
            accuracy_fn=accuracy_fn,
            loss_fn=loss_fn,
            device=device)

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=device)


total_train_time_model_0




# make prediction and get model 0 results
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
  # return a dict containing the results of model predicting on data loader.
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      # make our data device agnostic
      X, y = X.to(device), y.to(device)
      # make predictions
      y_pred = model(X)

      # accumulate the loss and accuracy values per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    # scale the loss and acc to find the avg loss and acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__,  # only works when model was created with a class 
          "model_loss": loss.item(),
          "model_acc": acc}




# get model_1 results dictionary
model_1_results = eval_model(model=model_1,
                             data_loader=test_dataLoader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
model_1_results





# Creating a convolutional neural network
class FashionMNISTModelV2(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    # print(x.shape)
    x = self.conv_block_2(x)
    # print(x.shape)
    x = self.classifier(x)
    return x
  


torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,   # here input_shape is of color_channels
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)



torch.manual_seed(42)

# create a batch of images
images = torch.randn(size=(32,3,64,64))
test_image = images[0]
images.shape





# create a single 2d layer
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

# pass the data through the convolution layer
conv_output = conv_layer(test_image)
conv_output.shape




# nn.MaxPool2d()

# create a sample nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# passing data through conv_layer
test_image_through_conv = conv_layer(test_image)
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# pass data through the ax pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layera and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")





# setup a loss function and optimizer for model_2
# from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                               lr=0.1)





torch.manual_seed(42)
torch.cuda.manual_seed(42)
from tqdm.auto import tqdm

# measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------------")
  train_step(model=model_2,
             data_loader=train_dataLoader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device = device)
  test_step(model=model_2,
            data_loader=test_dataLoader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)
train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)




model_2_results = eval_model(model=model_2,
                             data_loader=test_dataLoader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
model_2_results
