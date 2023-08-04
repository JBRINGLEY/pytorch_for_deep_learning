import sklearn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(f'first 5 samples of X:\n {X[0:5]}')
print(f'first 5 sample of Y:\n {y[0:5]}')

# make pandas df of data
circles_df = pd.DataFrame({'X1': X[:, 0],
                           'X2': X[:, 1],
                           "label": y})
print(circles_df.head())

# Visualize
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y)
# plt.show()

# Convert to tensors with default pytorch data type
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
print(X[:5], y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CircleModelv0(nn.Module):

    def __init__(self):
        super().__init__()
        # create 2 nn.LL capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


model_0 = CircleModelv0().to(device)
print(model_0)

## nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print(model_0)

# Make predictions with model
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f'Untrained Preds Shape: {untrained_preds.shape}')
print(f'\nFirst 10 untrained: {torch.round(untrained_preds[:10])}')

# Set up loss function and optimizer
# loss_fn = nn.BCELoss() # requires inputs go through sigmoid prior
loss_fn = nn.BCEWithLogitsLoss()  # sigmoid activtion fn built-in
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# Train model
# Going from raw logits -> prediction probabilities -> prediction labels
# can convert logits to prediction probabilities by passing them to activation func
# sigmoid for binary class and softmax for multiclass

y_logits = model_0(X_test.to(device))

# Use sigmoid function on model logits
y_pred_probs = torch.sigmoid(y_logits)
# get predicted labels
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))))
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
y_preds.squeeze()

# Optimization Loop
torch.manual_seed(42)
epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)  # BCEwithLogitsLoss expects logits
    acc = accuracy_fn(y_train, y_pred)
    # zero grad
    optimizer.zero_grad()
    # loss backward - backpropo
    loss.backward()
    # optimizer step
    optimizer.step()
    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # test loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch} | train loss: {loss:.2f} Acc: {loss:.2f} | test loss: {test_loss:.2f} test acc: {test_acc:.2f}')

# Make predictions and evaluate model
import requests
from pathlib import Path

# download help functions from Learn Pytorch if not already download
if Path("helper_functions.py").is_file():
    print('already exists')
else:
    print('downloading helper functions.py')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)


class CircleModelv1(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelv1().to(device)
# create loss fn and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

torch.manual_seed(42)
# Optimizer loop
epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_Test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    # forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # calculate loss
    train_loss = loss_fn(y_logits, y_train)
    train_acc = accuracy_fn(y_train, y_pred)
    # zero grad
    optimizer.zero_grad()
    # backprop
    train_loss.backward()
    # optimizer step
    optimizer.step()
    # test
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        # calculate loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | train loss: {train_loss:.2f} Acc: {train_acc:.2f} | test loss: {test_loss:.2f} test acc: {test_acc:.2f}')
