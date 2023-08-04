import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# Preparing and loading data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Visualize our data
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return:
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    # Plot test data
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

# plot_predictions()

class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))\

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# create random seed
torch.manual_seed(42)
# initialize lr class
model_0 = LinearRegressionModel()
# look at model paramters
print(list(model_0.parameters()))
print(model_0.state_dict())

## Making predictions using torch.inference_mode()
with torch.inference_mode(): # inference mode doesnt track gradients, can be useful if just making predictions
    y_preds = model_0(X_test)

print(y_preds)

# Setting up Loss Function and Optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)
# Build training loop
epochs = 200
epoch_count = []
train_loss_values = []
test_loss_values = []
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # trainmode: sets all parameters to require gradients
    # Forward pass
    y_pred = model_0(X_train)
    # Calculate loss
    loss = loss_fn(y_pred, y_train)
    # print(f'Loss: ', loss)
    # Optimzer zero grad
    optimizer.zero_grad()
    # Perform backpropogation on loss with respect to params
    loss.backward()
    # Step the optimizer
    optimizer.step()

    # Testing
    model_0.eval() # turns off different settings in the model not needed for evaluation
    with torch.inference_mode(): # turns off gradient tracking
        # forward pass
        test_pred = model_0(X_test)
        # calculate test loss
        test_loss = loss_fn(test_pred, y_test)

    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)


    # print(f'Epoch: {epoch} | Train Loss {loss} |Test Loss: {test_loss}')

print(model_0.state_dict())

plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label='Train Loss')
plt.plot(epoch_count, test_loss_values, label='Test Loss')
plt.title('Training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()


# Saving a PyTorch model
from pathlib import Path
# Create model dir
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# Create Model save path
MODEL_NAME = '01_pytorch_workflow_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(MODEL_SAVE_PATH)
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
# Load model
# to load in saved state dict, must initiate new class instance
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())
# Make some predictions with loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)


# Putting it all together, create device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# Data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split the data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Building a PyTorch Linear Model
class LinearRegressionModelv2(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionModelv2()
print(model_1, model_1.state_dict())

# Set the model to use the target device
model_1.to(device)
loss_fn = nn.L1Loss() # MAE
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)
torch.manual_seed(42)
epochs = 200
for epoch in range(epochs):
    model_1.train()
    # Forward pass
    y_pred = model_1(X_train)
    # Calculate loss
    loss = loss_fn(y_pred, y_train)
    # Optimizer zero grad
    optimizer.zero_grad()
    # Backpropogation
    loss.backward()
    # Optimizer step
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # Print whats happening
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Train Loss: {loss}, Test Loss {test_loss}')

print(model_1.state_dict())

# Making predictions
model_1.eval()
with torch.inference_mode():
    test_preds = model_1(X_test)

# Saving and loading
from pathlib import Path
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save model state dict
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

# Load Pytorch model
loaded_model_1 = LinearRegressionModelv2()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_1.to(device)
next(loaded_model_1.parameters())

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)