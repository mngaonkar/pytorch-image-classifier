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
| Linear  | 64\*64\*3  | 84 |
| Linear  | 84  | 50 |
| Linear | 50 | 3 |

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

