This was a 4 hour data coding challenge in the Kaggle style. The data happened to be NYPD crime data and I had to predict the type of crime (multiclass classification).

My approach was simple: I did some simple data cleaning, changed categorical variables to a one-hot encoding, and learned a gradient boosted machine (via lightgbm) over 3 folds. To predict on the test set, I took the average of the class probabilities over the 3 folds and predicted the highest probability.

My model had an accuracy of 49.2888%, compared to the baseline of 40.5600 (always predict grand larceny).

I was asked to answer these questions afterward:

**1. How would you improve your model if you had another hour?**

  First, I would apply Z-score normalization to the numerical input so it's all in a level statistical playing field. I'd probably have enough time to do some additional light feature selection. The lightgbm model has a "feature importance" attribute, I could use that to only train on the most important features. That is, I would actually prune the features rather than augment them. Some features are more helpful than others.

**2. How would you improve your model if you had another week?**

  Lots of things!
  - I would take more time to understand the data, explore it more. My goal in this challenge was more to throw the tricks I know at it quickly and so I didn't have time to think about the data itself.
  - Gather different models - random forests, neural networks, SVM, etc.
  - Gather subsets of the data - just the categories, just the numbers, counts of categories (instead of the categories themselves), etc. Also models trained on time subsets of the data - a model for just Friday, just October, etc.
  - Look at feature interactions (using correlation, information gain, ...) and experiment with combinations of features.
  - Blend all the models above (all combos of data subsets with all learning methods).
  The main idea would be to capture as many useful facets of the dataset as possible and rely on the blender to make use of all of them.
  - I would tune the base and blend models with cross validation.
