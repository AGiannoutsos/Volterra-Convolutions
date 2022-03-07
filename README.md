# Volterra-Convolutions
My thesis project on non linear Volterra Convolutions

## Abstract
Convolutional Neural Networks have caused a revolution in the field of computer vision in recent years, by continually breaking many state-of-the-art records.
These mathematical models are biologically-inspired and are derived from observations of the anatomy of the human visual cortex. They attempt to imitate its functions in a primitive manner with the use of the convolution operator.
Furthermore, research has shown the nature of non-linear processes within the human brain that contribute to the visual cognition process. The cellular structures of sensory neurons, as well as the ways in which information is transferred, lead to a non-linear correlation between the input signal and the final response.

However, since these CNNs are linear models, they are unable to adapt to the complicated nature of the visual perception.
As a response, several non-linear approaches, such as activation functions and, more crucially, the stacking of many convolutional layers, have been employed in practice. In recent years, research has focused on improving these arrangements so that the model can generalise with increasing flexibility on the data.
Although, there has been minimal study on improving the nature of the convolution process itself. 

To tackle this issue we introduce the Volterra convolutions.
Volterra series are polynomial approximation functions and are, in fact, the most well-known models for analyzing complex dynamic systems found in nature. 
As a result, they are deemed appropriate for enhancing the expressive capacity of the linear convolution operator, as well as introducing additional search spaces and dimensions for our estimation functions that are more susceptible to data variations. In other words, we can better replicate the human visual cortex this way and capture meaningful data correlations. 

In this study, we implement and evaluate the novel non-linear Volterra convolutions by using the CIFAR10 and CIFAR100 datasets.
We demonstrate that they outperform their linear counterparts with just minor changes to our model design.
Moreover, we cast light on how the information is interpreted and the higher level relations that arise in the receptive fields.
Also, we identify a resemblance between the non-linear terms of this method and the modern self-attention models that have contributed significant breakthroughs to the field of computer vision in the recent years.
Finally, we show relationships between the non-linear convolutions and the deeper layers of our network, revealing a resemblance to polynomial functions.

In this repository, the implementations of the non-linear convolutions can be found.

The complete experimentaion of the non-linear models together with checkpoints of the best performing models can be found in this [WandB project](https://wandb.ai/andreas_giannoutsos/Volterra-Convolutions).

## How to run an experiment

In order to run an experiment this repository needs to be cloned so that the Volterra convolution package together with the requirments are installed.

```git clone https://github.com/AGiannoutsos/Volterra-Convolutions.git
cd Volterra-Convolutions 
pip install -r requirements.txt
python Volterra-Convolutions/setup.py install
```

The experiments are preseted in the ./config directory. It is avised to use and [WanbB](https://wandb.ai/site) account and log in before conducting any experiment. However it is not needed and can be avoided.
As for example a running command could be:

```
python3 Volterra-Convolutions/configs/cifar100_volterra_n2_1.py
```




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




