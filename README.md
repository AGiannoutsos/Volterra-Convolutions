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

| Model         | CIFAR10 Accuracy          | Mean         | 
|:-------------:|:-------------------------:|:------------:|
| Linear        |   95.39, 95.21, 95.20     |   95.26      | 
| Volterra 2nd  |   95.22, **95.50**, 95.44 |   **95.38**  |
| Volterra 3rd  |   95.31, 95.20, 95.39     |   95.30      |

| Model         | CIFAR100 Accuracy  | Mean         | 
|:-------------:|:------------------:|:------------:|
| Linear        |   **77.91**, 77.74 |   **77.50**  | 
| Volterra 2nd  |   77.57, 77.19     |   77.38      |
| Volterra 3rd  |   77.64, 76.93     |   77.28      |

### Grid Search setup

#### Best Parameters Grid Search
| Parameter          | Values      |
|:------------------:|:-----------:|
| Init channel       |   16, 160   |
| Kernel size        |   3, 5      |
| Dilation           |   1, 2, 3   |


#### Best Scaling Parameters Grid Search
| Parameter          | Values        |
|:------------------:|:-------------:|
| Scaling            |   True, False |
| Masking            |   True, False |

### Grid Search results


#### Best Parameters Grid Search
| Best Model         | Init channel | Kernel size | Dilation | CIFAR10 Accuracy |
|:------------------:|:------------:|:-----------:|:--------:|:----------------:|
| Volterra 2nd       |   16         | 3           | 2        | **95.53**        |


#### Best Scaling Parameters Grid Search
| Best Model         | Scaling | Masking | CIFAR10 Accuracy |
|:------------------:|:-------:|:-------:|:----------------:|
| Volterra 2nd       | True    | False   | **95.60**        |




