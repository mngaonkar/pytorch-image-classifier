# Image Classifier in PyTorch

Basic image classifier for identifying multiple classes

## Hyper parameters for training simple and convolution neural network
| Hyper Parameter  | Value |
| ------------- | ------------- |
| Epochs  | 30  |
| Batch size  | 32  |
| Learning rate | 0.001 |

## Simple neural network
| Layer  | In Features | Out Features |
| ------------- | ------------- | ------------- |
| Linear  | 64 x 64 x 3  | 84 |
| Linear  | 84  | 50 |
| Linear | 50 | 3 |

## Convolution neural network
| Layer  | Description |
| ------------- | ------------- | 
| Conv2D  | In Features = 3, Out Features = 12, Kernel Size = 3, Stride = 1, Padding = 1  |
| ReLU  |   | 
| Conv2D  | In Features = 12, Out Features = 12, Kernel Size = 3, Stride = 1, Padding = 1  |
| ReLU  |   | 
| MaxPool2D | Size = 2 x 2 |
| Conv2D  | In Features = 12, Out Features = 24, Kernel Size = 3, Stride = 1, Padding = 1  |
| ReLU  |   | 
| Conv2D  | In Features = 24, Out Features = 24, Kernel Size = 3, Stride = 1, Padding = 1  |
| ReLU  |   | 
| Linear | In Features = 32 x 32 x 24, Out Features = 3 |

## Training accuracy with simple neural network

| Result  | Value |
| ------------- | ------------- |
| Training accuracy  | 0.853961  |
| Validation accuracy  | 0.747073  |


## Training accuracy with convolution neural network

| Result  | Value |
| ------------- | ------------- |
| Training accuracy  | 0.854475  |
| Validation accuracy  | 0.838407  |

