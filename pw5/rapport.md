# Introduction

This project involves building an app to identify duck species using machine learning. 
We've gathered a diverse set of images from the web using a Python-based web scraper. 
To broaden our dataset, we applied data augmentation techniques like rotation and scaling. 
For processing, we're using Convolutional Neural Networks (CNNs) and transfer learning, which allow efficient and effective species recognition. 
This tool has potential uses in areas like ornithology and wildlife conservation.

# The problem

We are trying to learn to differentiate eight unique classes of ducks, comprised of seven species of male ducks and one specie of female ducks. 
The focus on male ducks arises from their vibrant and distinct coloration, in contrast to the more uniform brownish and greyish hues of females.

The dataset collected for this project is very balanced as all the classes have the same amount of images. 
For each of the eight classes, we have a total of 50 images (40 used for training and 10 for testing).
This amounts of 400 images in total, a manageable amount for detailed processing and analysis.

| Duck Species            | Training Images | Testing Images |
|-------------------------|-----------------|----------------|
| Allier White Duck       | 40              | 10             |
| Gadwall Duck            | 40              | 10             |
| Mallard Duck            | 40              | 10             |
| Mandarin Duck           | 40              | 10             |
| Northern Shoveler Duck  | 40              | 10             |
| Tufted Duck             | 40              | 10             |
| Whistling Duck          | 40              | 10             |
| Female Goosander        | 40              | 10             |


/////////////////////////// TODO AJOUTER GRAPHE ///////////////////////////

The intra-class diversity is relatively low, as all ducks within a specific class exhibit similar characteristics.
However, the inter-class similarity is also quite low. Each species of duck we have chosen exhibits unique color patterns, reducing the chances of misclassification between classes.
These two factors should allow the CNN to effectively learn the distinguishing features of each class and accurately identify the species of ducks in new unseen images.

# Data Preparation

The initial step in our data preparation involved resizing and rescaling the images.
To ensure compatibility with our model, we resized all images to a standard dimension of 224x224 pixels.

To enhance our dataset, we implemented data augmentation.
These techniques included transformations such as rotation, zooming, etc...
This process generated additional training samples.

Our dataset was split into two subsets: training and testing. 
We allocated 40 images from each class to the training set and the remaining 10 images to the test set, achieving an 80-20 split. 

# Model Creation

We decided to compare four different models to determine which one would be the most effective for our task.
The difference between the models are the variation of epochs and neurons per layer.
All the other parameters were constant for all the models because they were the most effective for our task.

## Hyperparameters exploration

The four models are the following:

- Model 1: 10 epochs, 100 neurons per layer
- Model 2: 10 epochs, 200 neurons per layer
- Model 3: 20 epochs, 100 neurons per layer
- Model 4: 20 epochs, 200 neurons per layer

/////////////////////////// TODO AJOUTER GRAPHES ///////////////////////////

The model with the best accuracy was model X.

## Final model parameters

The final model we chose was model X had these parameters :

- X epochs
- X neurons per layer
- X layers
- X batch size
- X learning rate
- X optimizer
- X activation function
- X loss function

## Transfer learning

We performed the transfer learning by freezing all the layers except the last one. 
We used transfer learning because it is a very effective technique to train a model with a small dataset. 
The model we used for transfer learning is the /// TODO AJOUTER NOM DU MODELE /// model.
With this method, we had a very good model without having to train it for a long time and this is the main advantage of transfer learning.

# Results

## Confusion matrix

## F-score for each class

## Results after model evaluation

## Grad-cam analysis

## Misclassified images

## Dataset improvement

## Classes confusion


# Conclusion

