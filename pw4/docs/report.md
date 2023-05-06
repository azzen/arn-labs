---
title: "Deep Neural Networks"
author: "ANNEN Rayane, MARTINS Alexis"
date: "12.04.2023"
subtitle: "Apprentissage par réseaux de neurones artificiels"
lang: "en"
titlepage: true
titlepage-logo: ./figures/HEIG-Logo.png
titlepage-rule-color: "DA291C"
toc: true
toc-own-page: true
header-includes:
    - \usepackage{float}
    - \usepackage{subfig}

---

# Digit recognition from raw data

__What is the learning algorithm being used to optimize the weights of the neural
networks? What are the parameters (arguments) being used by that algorithm? What
cost function is being used ? please, give the equation(s)__

The algorithm used to optimize the weight is RMSprop (Root Mean Square Propagation). 

The parameters are the following:

```py
tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=100,
    jit_compile=True,
    name="RMSprop",
    **kwargs
)
```

The following equations are used: 
\begin{align*}
    E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)\left(\frac{\partial C}{\partial w}\right)^2 \\
    w_t = w_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t}}\frac{\partial C}{\partial w}
\end{align*}

where $\eta$ is the learning rate, $w_t$ the new weight, $\beta$ is the moving average parameter, $E[g]$ is the moving average of squared gradients and $\frac{\partial C}{\partial w}$ is the derivative of the cost function with respect to the weight.

The cost function is the categorical cross-entropy loss function:

$$
    \text{CE} = -\frac{1}{N} \sum_{k=0}^{N} \log \vec{p_i}[y_i]
$$

where $N$ is the number of samples, $\vec{p_i}$ is the neural network output and $y_i$ is the target class index.

## Shallow Neural Network

For this experiment a simple shallow neural network is used. We use raw data to classify the digits.

### Hyper-parameters

Changed made to the model from the original: we reduced the number of neurons in the hidden layer from 300 to 100.

- Epochs: 10
- Hidden layers:
  - 100 neurons, reLU
- Output activation function: softmax.
- Batch size: 128

- Weights in the hidden layer: 784 * 100 + 100 = 78400 + 100 = 78500
- Weights in the output layer = 10 * 100 + 10 = 1010
- Total weights: 78500 + 1010 = 79510

# Digit recognition from features of the input data

## Shallow Neural Network

In this experiment, we use the Histogram of gradients (HOG) features to classify the digits.

### Hyper-parameters

HOG: 

 - orientation count: 8
 - pixels per cell: 4

- Epochs: 10
- Hidden layers:
  - 100 neurons, reLU
- Output activation function: softmax.
- Batch size: 128

- Weights in the hidden layer: 392 * 200 + 200 = 78400 + 200 = 78600
- Weights in the output layer = 10 * 200 + 10 = 2010
- Total weights : 78600 + 2010 = 80610

# Convolutional neural network digit recognition

## Deep Convolutional Neural Network

- Epochs: 10
- Batch size: 128

- Hidden layers:
  - Convolutional 2D: 5x5 
  - MaxPooling 2D: pool size: 2x2
  - Convolutional 2D: 5x5
  - MaxPooling 2D: pool size: 2x2
  - Convolutional 2D: 3x3
  - MaxPooling 2D: 2x2
  - Flatten layer
  - Dense (25 neurons), activation function: reLU

Output: 10 neurons, activation function: softmax

Model complexity :

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 l0 (InputLayer)             [(None, 28, 28, 1)]       0         
                                                                 
 l1 (Conv2D)                 (None, 28, 28, 9)         234       
                                                                 
 l1_mp (MaxPooling2D)        (None, 14, 14, 9)         0         
                                                                 
 l2 (Conv2D)                 (None, 14, 14, 9)         2034      
                                                                 
 l2_mp (MaxPooling2D)        (None, 7, 7, 9)           0         
                                                                 
 l3 (Conv2D)                 (None, 7, 7, 16)          1312      
                                                                 
 l3_mp (MaxPooling2D)        (None, 3, 3, 16)          0         
                                                                 
 flat (Flatten)              (None, 144)               0         
                                                                 
 l4 (Dense)                  (None, 25)                3625      
                                                                 
 l5 (Dense)                  (None, 10)                260       
                                                                 
=================================================================
Total params: 7,465
Trainable params: 7,465
Non-trainable params: 0
_________________________________________________________________
```

# Experiments

## Raw data

### 100 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_100neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_100neurones_m.png}
  }}%
  \caption{Model with raw data and 100 neurons in the dense layer}

\end{figure}


### 300 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_300neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_300neurones_m.png}
  }}%
  \caption{Model with raw data and 300 neurons in the dense layer}

\end{figure}


### 600 neurones

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_600neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/RAW_600neurones_m.png}
  }}%
  \caption{Model with raw data and 600 neurons in the dense layer}

\end{figure}


We can notice that the first model is indeed the most suitable for this situation. The more we increase the number of neurons, the more it overfits and the results aren't better.

An overall observation of the models is they often confuse the 9 with the 4. For the last two, they also have a tendency to confuse the 5 with the 3.

## Features-based (HOG)

### PIX_P_CELL 4, orientation 8, 100 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori8_100neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori8_100neurones_m.png}
  }}%
  \caption{Model using HOG features with 4 pixels per cell, 8 orientations and 100 neurons}

\end{figure}


### PIX_P_CELL 4, orientation 8, 300 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori4_100neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori4_100neurones_m.png}
  }}%
  \caption{Model using HOG features with 4 pixels per cell, 4 orientations and 100 neurons}

\end{figure}


### PIX_P_CELL 4, orientation 4, 100 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori8_300neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix4_ori8_300neurones_m.png}
  }}%
  \caption{Model using HOG features with 4 pixels per cell, 8 orientations and 300 neurons}

\end{figure}

### PIX_P_CELL 7, orientation 8, 100 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix7_ori8_100neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/HOG_pix7_ori8_100neurones_m.png}
  }}%
  \caption{Model using HOG features with 7 pixels per cell, 8 orientations and 100 neurons}

\end{figure}

For us the best model is the seconde one. Compared to the others, it's one of the best in terms of performances and it doesn't seem to overfit. Models (1) and (3) are totally overfitting and the performances are not a lot better than (2). The last model is also very good, but the loss is higher than the second.

For this set of models, the number 9 is clearly the worse. It often is confused as other numbers (like 3 and 4) or other numbers are confused as a 9 (like 4, 7 and 8).

## CNN

### 25 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_25neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_25neurones_m.png}
  }}%
  \caption{Model using CNN 25 neurons in the feed forward part}
\end{figure}

### 250 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_250neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_250neurones_m.png}
  }}%
  \caption{Model using CNN 250 neurons in the feed forward part}
\end{figure}

### 500 neurons

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_500neurones_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN_500neurones_m.png}
  }}%
  \caption{Model using CNN 500 neurons in the feed forward part}

\end{figure}

Once again the model with less neurons is the best one. Not necessarily in the performances displayed by the loss function, it's slightly higher than the rest. But at least it doesn't overfit like the two models remaining.

In this set, the confusion is more spread. We can't see a particular number with a lot of failures in all the models.

# Custom model with convolutional deep neural networks

__Train a CNN to solve the MNIST Fashion problem, present your evolution of the errors during training and perform a test. Present a confusion matrix, accuracy, F-score and discuss your results. Are there particular fashion categories that are frequently confused?__


## Experiments

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_15neurones_20ep_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_15neurones_20ep_m.png}
  }}%
  \caption{Model using CNN 15 neurons in the feed forward part and 20 epochs}

\end{figure}

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_10neurones_10ep_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_10neurones_10ep_m.png}
  }}%
  \caption{Model using CNN 10 neurons in the feed forward part and 10 epochs}

\end{figure}

\begin{figure}[H]
  \centering
  \subfloat[\centering Error graph]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_30neurones_20ep_g.png}
  }}%
  \qquad
  \subfloat[\centering Confusion matrix]{
  \scalebox{0.45}{%
    \includegraphics{./figures/CNN2_30neurones_20ep_m.png}
  }}%
  \caption{Model using CNN 30 neurons in the feed forward part and 20 epochs}

\end{figure}

We notice that all the models are not very good. The best one is probably the last because it didn't have time to start overfitting like the others. The performances is clearly not the best.

All the models tends to confuse tops (t-shirts, shirts, etc.).


# General questions

__Do the deep neural networks have much more “capacity” (i.e., do they have more
weights?) than the shallow ones? explain with one example__

Yes, deep neural networks generally have much more capacity than shallow ones, as they have more layers and consequently more weights. 

Counterintuitively, it is observed that the shallow model has many more parameters than the deep model. This is because a shallow model is heavily interconnected, which increases the number of parameters.

We can use the example of the laboratory where the shallow model has 10 times more parameters than the deep one.