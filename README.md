# k-Nearest-Neighbors-algorithm
In this project, I implemented the k-Nearest Neighbors (k-NN) algorithm from scratch to classify mushrooms as poisonous or edible based on given features. I was only allowed to use NumPy, Pandas, and Matplotlib, without relying on machine learning libraries like Sklearn. The project also involved cross-validation, hyperparameter tuning, and ROC curve analysis to evaluate and optimize the classifier.

First, I conducted data preprocessing, ensuring that the dataset was cleaned and properly normalized, as k-NN is sensitive to scale differences. I then implemented the standard k-NN classifier, using the Euclidean distance metric and setting k to the square root of the training set size. I tested the model on a separate test set and reported F1-score, accuracy, precision, and recall.

Next, I modified the k-NN algorithm to use a threshold-based classification instead of majority voting. If at least 40% of the neighbors were classified as poisonous, the mushroom was predicted as poisonous. I compared the results of this modified method against the original majority-voting k-NN, analyzing changes in precision and recall.

For hyperparameter tuning, I performed 5-fold cross-validation to optimize the choice of k and distance metric. I tested at least five values of k and experimented with three distance metrics (including Manhattan and Cosine similarity). I plotted F1-score curves to visualize performance across different values and reported the best combination of parameters.

Finally, I evaluated the classifier using ROC curve analysis. I plotted the true positive rate (TPR) vs. false positive rate (FPR) at different threshold levels to visualize model performance. Additionally, I computed the Area Under the Curve (AUC) from scratch to quantify the classifier’s effectiveness.

This project helped reinforce my understanding of k-NN, cross-validation, threshold-based classification, and performance evaluation using ROC and AUC metrics.

# Decision Tree

In this project, I implemented and analyzed decision tree models for credit risk classification using the German Credit Risk dataset. The task involved predicting whether a person is credit-worthy based on demographic and financial characteristics. The dataset was preprocessed with one-hot encoding for categorical variables and split into training and test sets.

First, I constructed a basic decision tree classifier using sklearn.tree.DecisionTreeClassifier. I performed 5-fold cross-validation on the max_depth parameter (ranging from 1 to 10) and evaluated the model using accuracy, F1-score, and AUC. By analyzing the results, I determined the most appropriate depth for optimal performance.

Next, I fine-tuned the decision tree parameters using GridSearchCV with 5-fold cross-validation. I optimized max_depth along with at least two other hyperparameters, creating a grid of 27 parameter combinations. After identifying the best hyperparameter set, I trained a final decision tree model and plotted the tree using sklearn.tree.plot_tree. The model’s F1-score, accuracy, and AUC were reported for comparison.

To further optimize the model, I applied Generalized and Scalable Optimal Sparse Decision Trees (GOSDT). I tuned at least one parameter with three distinct values using 5-fold cross-validation, ensuring the ‘balance’ setting was enabled to account for class imbalance. The best GOSDT model was then retrained on the full training set, and its F1-score, accuracy, and AUC were evaluated on the test set.

Finally, I conducted a model comparison, analyzing which decision tree configuration—basic, fine-tuned, or GOSDT—performed best for this dataset. The project provided insights into the trade-offs between model complexity, interpretability, and performance in credit risk classification.
