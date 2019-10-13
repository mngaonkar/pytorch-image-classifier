import torch
import torchvision
from torchvision import transforms, models
import torch.optim as optim
from NeuralNetwork import NeuralNetwork
from ConvNeuralNetwork import ConvNeuralNetwork
from torch.utils.data import DataLoader
from PIL import ImageFile
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True

# model = NeuralNetwork(num_classes=3)
# model = ConvNeuralNetwork(num_classes=3)
model = models.resnet18(pretrained=True)

# Configuration
training_data_path = "./train/"
validation_data_path = "./validation"
test_data_path = "./test"

# Hyper parameters
batch_size = 32
epochs = 30
learning_rate = 0.001

# Transform input images
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])
])

# Load traing and validation data
train_data = torchvision.datasets.ImageFolder(root=training_data_path, transform=train_transforms)
validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transforms)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms)

print("classes = ", train_data.classes)

training_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
validation_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()

# Train model
training_loss = []
training_accuracy = []
training_index = 0
total = 0
correct = 0

for epoch in range(epochs):
    training_index = 0

    for index, data in enumerate(training_data_loader):
        inputs, labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        training_accuracy.append(correct/total)

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.data.item())
        training_index += 1
        print("epoch = %d training accuracy = %f training loss = %f" % (epoch, correct/total, loss.data.item()))

    print("Number of images processed = %d" % (training_index * len(inputs)))

# print("training index = ", training_index)
# print("training loss = ", training_loss/training_index)
# plt.plot(training_accuracy, label='Training Accuracy')
# plt.plot(training_loss, label='Training Loss')
# plt.legend(frameon=False)
# plt.show()

# print(net.eval())

# Evaluate model
validation_loss = 0
validation_index = 0

total = 0
correct = 0

model.eval()

for index, data in enumerate(validation_data_loader):
    inputs, labels = data
    outputs = model(inputs)
    # print(outputs)
    _, predicted = torch.max(outputs, -1)
    # print(predicted)
    # print(labels)
    total += labels.size(0)
    # print("total images count = %d" % total)
    # print(labels)
    correct += (predicted == labels).sum().item()
    # print("correct prediction count = %d" % correct)
    loss = loss_fn(outputs, labels)
    validation_loss += loss.data.item()
    validation_index += 1
    # print("validation loss = %f" % loss.data.item())

print("validation loss = ", validation_loss/validation_index)
print("validation accuracy = %f" % (correct/total))

# Save weights
torch.save(model, 'model.pth')

# Prediction
labels = train_data.classes

image = Image.open("./test/burger/burger-close-up-delicious-1639565-620x370.jpg")
image = test_transforms(image)
image = image.unsqueeze(0)

prediction = model(image)
print("prediction index = ", torch.max(prediction, -1))
print("real = [burger] prediction = ", labels[torch.argmax(prediction)])

image = Image.open("./test/pizza/pizza-saucisse-piquante-2301.jpg")
image = test_transforms(image)
image = image.unsqueeze(0)

prediction = model(image)
print("prediction index = ", torch.max(prediction, -1))
print("real = [pizza] prediction = ", labels[torch.argmax(prediction)])