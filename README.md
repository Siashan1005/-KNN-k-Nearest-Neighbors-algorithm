# -KNN-k-Nearest-Neighbors-algorithm
In this project, I implemented the k-Nearest Neighbors (k-NN) algorithm from scratch to classify mushrooms as poisonous or edible based on given features. I was only allowed to use NumPy, Pandas, and Matplotlib, without relying on machine learning libraries like Sklearn. The project also involved cross-validation, hyperparameter tuning, and ROC curve analysis to evaluate and optimize the classifier.

First, I conducted data preprocessing, ensuring that the dataset was cleaned and properly normalized, as k-NN is sensitive to scale differences. I then implemented the standard k-NN classifier, using the Euclidean distance metric and setting k to the square root of the training set size. I tested the model on a separate test set and reported F1-score, accuracy, precision, and recall.

Next, I modified the k-NN algorithm to use a threshold-based classification instead of majority voting. If at least 40% of the neighbors were classified as poisonous, the mushroom was predicted as poisonous. I compared the results of this modified method against the original majority-voting k-NN, analyzing changes in precision and recall.

For hyperparameter tuning, I performed 5-fold cross-validation to optimize the choice of k and distance metric. I tested at least five values of k and experimented with three distance metrics (including Manhattan and Cosine similarity). I plotted F1-score curves to visualize performance across different values and reported the best combination of parameters.

Finally, I evaluated the classifier using ROC curve analysis. I plotted the true positive rate (TPR) vs. false positive rate (FPR) at different threshold levels to visualize model performance. Additionally, I computed the Area Under the Curve (AUC) from scratch to quantify the classifier’s effectiveness.

This project helped reinforce my understanding of k-NN, cross-validation, threshold-based classification, and performance evaluation using ROC and AUC metrics.
