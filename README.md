# Volterra-Convolutions
My thesis project on non linear Volterra Convolutions


## Training setup

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |    0.9   |   true   |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |    0.9   |   true   |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |    0.9   |   true   |
| 200 ~ 220 |     0.00016   |     0.0       | Momentum  |    0.9   |   true   |

## Experiment Results

### Volterra results

| Model              | CIFAR10 | CIFAR100 | 
|:------------------:|:-------:|:--------:|
| Linear 28x10       |   0-0   |   0-0    | 
| Volterra 2nd 28x10 |   0-0   |   0-0    |
| Volterra 3rd 28x10 |   0-0   |   0-0    |

### Grid Search setup

| Parameter          | Values      |
|:------------------:|:-----------:|
| Init channel       |   16, 160   |
| Kernel size        |   3, 5, 7   |
| Dilation           |   1, 2, 3   |

### Grid Search results
