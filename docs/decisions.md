DSC-0001: Decision Tree Classifier
==================================

**Date**: 2024-11-09  
**Decision**: Implement a DecisionTreeClassifier model for classification tasks.  
**Status**: Accepted  

**Motivation**:  
The DecisionTreeClassifier was chosen due to its simplicity and flexibility, as it can handle mixed data types and has no assumptions about data distribution.

**Reason**:
Decision Trees provide a good compromise between simplicity and functionality for machine learning tasks.

**Limitations**: 
Not as robust as other methods on complex datasets and sensitive to data imbalances.

**Alternatives**:
Naive Bayes, Neural Networks

---

DSC-0002: K-Nearest Neighbors (KNN)
===================================

**Date**: 2024-11-09  
**Decision**: Implement a K-Nearest Neighbors model for classification tasks.  
**Status**: Accepted  

**Motivation**:  
The project requires a simple, non-parametric classification algorithm capable of handling multi-class problems.

**Reason**:  
KNN works well for smaller datasets and does not make strong assumptions about the data distribution. It uses distance metrics to classify data points, which makes it versatile across different types of data.

**Limitations**:  
Computationally expensive for large datasets, as predictions involve comparing the input to all stored instances.

**Alternatives**:  
Support Vector Machines (SVM)

---

DSC-0003: Random Forest Classifier
==================================

**Date**: 2024-11-09  
**Decision**: Implement a Random Forest Classifier model for classification tasks.  
**Status**: Accepted  

**Motivation**:  
The project requires a robust and versatile classification algorithm that can handle high-dimensional datasets, avoid overfitting, and perform well with minimal hyperparameter tuning.

**Reason**:  
Random Forest is an ensemble learning method that aggregates the predictions of multiple decision trees to improve classification performance. It is particularly effective in preventing overfitting, even with large and complex datasets.

**Limitations**:  
Prediction time can be slower compared to simpler models, as it involves aggregating results from multiple trees.

**Alternatives**:  
Decision Tree Classifier, Gradient Boosting Machines

---

DSC-0004: ARD Regression
========================

**Date**: 2024-11-09  
**Decision**: Implement an ARD Regression model for regression tasks.  
**Status**: Accepted  

**Motivation**:  
ARD Regression is ideal for high-dimensional data as it automatically identifies and discards irrelevant features, reducing overfitting.

**Reason**:  
ARD Regression uses Bayesian inference to assign relevance to each feature, providing a robust, interpretable model that automatically handles feature selection.

**Limitations**:  
Sensitive to outliers.

**Alternatives**:  
Linear Regression, Elastic Net, Lasso Regression

---

DSC-0005: Lasso Regression
==========================

**Date**: 2024-11-09  
**Decision**: Use a Lasso Regression model for regression tasks.  
**Status**: Accepted  

**Motivation**:  
Lasso Regression helps with feature selection and regularization, addressing overfitting in high-dimensional datasets by shrinking some coefficients to zero.

**Reason**:  
Lasso is a linear model that performs both regularization and feature selection. By penalizing the absolute value of coefficients, it encourages sparsity, leading to simpler, more interpretable models.

**Limitations**:  
Can struggle with highly correlated features, as it may arbitrarily select one feature over another.

**Alternatives**:  
Ridge Regression, Elastic Net

---

DSC-0006: Ridge Regression
==========================

**Date**: 2024-11-09  
**Decision**: Use a Ridge Regression model for regression tasks.  
**Status**: Accepted  

**Motivation**:  
Ridge Regression is used to prevent overfitting in linear regression models by adding an L2 penalty to the coefficients, making the model less sensitive to multicollinearity.

**Reason**:  
Ridge regression reduces the model's complexity by penalizing large coefficients, thus improving the model's robustness and accuracy.

**Limitations**:  
It can be ineffective when there are irrelevant features in the dataset.

**Alternatives**:  
Lasso Regression, Elastic Net


DSC-0007: Metrics Implementation
==========================

**Date**: 2024-10-20  
**Decision**: Implement metrics for model evaluation.
**Status**: Accepted  

**Motivation**:  
To evaluate the performance of machine learning models.

**Reason**:  
Each metric provides a quantitative measure of model accuracy, precision, or error.

The following formulas were used for the metrics:
## 1. **Mean Squared Error (MSE)**

$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

- $\( y_i \)$ = ground truth value (true value)
- $\( \hat{y}_i \)$ = predicted value
- $\( n \)$ = number of samples

## 2. **Mean Absolute Error (MAE)**

$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

- $\( y_i \)$ = ground truth value
- $\( \hat{y}_i \)$ = predicted value
- $\( n \)$ = number of samples

## 3. **R Squared (R²)**

$R^2 = (\text{corr}(y, \hat{y}))^2$

- $\( \text{corr}(y, \hat{y}) \)$ is the Pearson correlation coefficient.

## 4. **Accuracy**

$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{\sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i)}{n}$

- $\( y_i \)$ = ground truth label
- $\( \hat{y}_i \)$ = predicted label
- $\( n \)$ = number of samples
- $\( \mathbb{1} \)$ is an indicator function that is 1 if $\( y_i = \hat{y}_i \)$, otherwise 0.

## 5. **Micro Precision**

$\text{Micro Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$

- **True Positives (TP)**: The number of positive instances correctly predicted as positive.
- **False Positives (FP)**: The number of negative instances incorrectly predicted as positive.

## 6. **Micro Recall**

$\text{Micro Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$

- **True Positives (TP)**: The number of positive instances correctly predicted as positive.
- **False Negatives (FN)**: The number of positive instances incorrectly predicted as negative.

**Limitations**:  
Some metrics, like R_square, assume specific data distributions.

**Alternatives**:  
Use external libraries such as scikit-learn.


DSC-0008: Artifacts ID
==========================

**Date**: 2024-10-25  
**Decision**: Use a base64-encoded combination of asset_path and version to generate a unique artifact ID.
**Status**: Accepted  

**Motivation**: Using a combination of asset_path and version ensures that each artifact can be uniquely identified, even if artifacts share the same name but differ in version or location.

**Reason**: Base 64 encoding provides uniqueness, non-colliding format, making retrieval and matching easier.

**Limitations**: Base64 encoding increases the length of the ID, which may slightly impact storage.

**Alternatives**: Concatenation without encoding, Hashing.


DSC-0009: Execute Method Pipeline
==========================

**Date**: 2024-10-25  
**Decision**: Implement the execute method to run the complete ML pipeline.
**Status**: Accepted  

**Motivation**: To streamline and automate the machine learning workflow, ensuring consistent preprocessing, training, and evaluation.

**Reason**: After evaluating on the test set, this method temporarily switches to evaluating on the training set to calculate training metrics, allowing a comparison between training and testing performance. The test data is then restored for any subsequent operations.
 A unified method simplifies the pipeline's operation, reduces manual errors, and provides a standard structure for metrics and predictions. (lines 228-290 in pipeline)
 
**Limitations**: Metrics may not fully capture model behavior on unseen data if the dataset is small.

**Alternatives**: Putting arguments in the evaluate function.

DSC-0010: Dataset getter and setter inside pipeline class
==========================

**Date**: 2024-10-25  
**Decision**: Implement a getter and a setter.
**Status**: Accepted  

**Motivation**: In the deployment page, the dataset attribute had to be set after the pipeline was created. 
It was necessary to validate the value of this dataset since the input and target features in the pipeline had to be the same as the ones in the uploaded dataset.

**Reason**: Validation and leakage prevention needed.

**Alternatives**: Reinstantiate the pipeline with the selected dataset.

DSC-0011: Metrics getter
==========================

**Date**: 2024-10-25  
**Decision**: Implement a getter for _metrics attribute in the Pipeline class.
**Status**: Accepted  

**Motivation**: On the Modelling page, the metrics need to be put in the serialized data dictionary containing the other pipeline components and, to that end, we need to safely access the list containing them.

**Reason**: Leakage prevention.


DSC-0012: Split values
==========================

**Date**: 2024-10-25  
**Decision**: What values the split can take.
**Status**: Accepted  

**Motivation**: The KNN k value is 5 by default. Giving that the minimum value for split was 0.01, there wouldn't be enough neighbors for a prediction if a dataset had less than 500 samples.
Therefore, to prevent any problems, we made the minimum split value 0.2 and it increases by 0.05.

**Reason**: The KNN model cannot give a prediction when the split is very small.
 
**Limitations**: 

**Alternatives**: Add warning message specifically for KNN predictions, to encourage user to choose a larger split value.


DSC-0013: Dictionaries with artifact names as keys and artficacts as values
==========================

**Date**: 2024-10-25  
**Decision**: Create dictionaries with artifact names as keys and artficacts as values, in the Modelling and Deployment pages.
**Status**: Accepted  

**Motivation**: We wanted the user to be prompted with the artifacts name, not the artifacts themselves.

**Reason**: For clarity reasons.
 
**Limitations**: Storage, since dictionaries take up more storage.

**Alternatives**: Tuples with the same functionality.
