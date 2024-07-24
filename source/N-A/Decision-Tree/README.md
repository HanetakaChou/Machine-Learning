# Decision Tree

## Classification Tree

### Binary Tree

First, we would like to discuss the situation where each feature has only two possible values (namely, **binary**).

For the **binary classification**, let $\displaystyle p_1$ be the fraction of the **training examples** which belong to the class and $\displaystyle p_0$ be the fraction of the **training examples** which do NOT belong to the class. Evidently, we have that $\displaystyle p_0 + p_1 = 1$.  

We use the **entropy** $\displaystyle \mathop{\mathrm{H}}(p_1) = - (p_1 \log_2 p_1 + (1 - p_1) \log_2 (1 - p_1) )$ to the measure the **impurity** of the tree node. Evidently, the **entropy** $\displaystyle \mathop{\mathrm{H}}(p_1)$ reaches the maximum value of 1 when $\displaystyle p_1 = \frac{1}{2}$.  

It should be noted that the **entropy** should be distinguished form the **cross entropy** in the **logistic regression**.  

The **cross entropy** is defined as $\displaystyle \mathop{\mathrm{H}}(p_1, q_1) = - (p_1 \ln q_1 + (1 - p_1) \ln (1 - q_1) )$ of which $\displaystyle p_1$ is the **target** and $\displaystyle q_1$ is the **prediction**. But we do NOT have $\displaystyle q_1$ for the **entropy**. Another difference is that we use the $\displaystyle \ln$ for the **cross entropy** while we use the $\displaystyle \log_2$ for the **entropy**.  

At the beginning, all **training examples** are stored in the **root node**. And then we use the **recursion** to split the **training examples** in the parent node into the left node and the right node.  

When we split, we try to maximize the purity, namely, minimize the impurity (namely, the **entropy**). We calculate the **entropy** of the old parent node and the new left node and the new right node. The reduction of the **entropy** $\displaystyle \mathop{\mathrm{H}}(p_1^{parent}) - (W^{left} \mathop{\mathrm{H}}(p_1^{left}) + W^{right} * \mathop{\mathrm{H}}(p_1^{right}))$, of which the weights $\displaystyle W^{left}$ and $\displaystyle W^{right}$ are the fraction of the **training examples** in the left node and the right node, is called the **information gain** which means the improvement of the purity. We choose the **feature** with the highest **information gain** to split, and the **recursion** will stop when the highest **information gain** is below the preset threshold. (When all **training examples** belong to the same class, the highest **information gain** can NOT be greater than zero, and the **recursion** should stop.)  

There are usually some other criteria to stop the **recursion**. For example, the preset maximum depth, the preset minmum number of the training examples in the node.  

It should be noted that the depth of the binary tree can usually reflect the model complexity. The preset maximum depth can hopefully avoid **overfitting**.  

### One-Hot Encoding  

When the **categorical** (namely, discrete) **feature** has more than two possible values, we can use the **one-hot encodeing** to convert this feature into multiple features with only two possible values.  

This technique can also be used to convert the **multiclass classification** (**multinomial logstic regression**) into the  **multi-label classification** (**logstic regression**).   

### Continuous Valued Features  

We use the simple equation $\displaystyle y < \text{threshold}$ to split the **training examples** into the left node and the right node.  

Evidently, this simple equation can NOT compete with the **linear hyperplane decision boundary** in the **logstic regression**.  

One convention is to first sort the values of all training examples and then calculate the **information gain** for all mid-points between these values and select the highest.  

## Regression Tree

The regression tree is to predict a number instead of a label (namely, the **classification**).  

The difference is that we try to calculate the the **variance** instead of the **entropy**. And the variance is still weighted by the fraction of the number of the **training examples** $\displaystyle \sigma_y^{parent} - (W^{left} \sigma_y^{left} + W^{right} \sigma_y^{right})$.  

We use the average value of all **training examples** in the leaf node as the **prediction**.  

## Tree Ensembles  

### Bagged Decision Tree  

Even the change of one **training example** can make the **recursion** to choose another different split at the root node and create another totally different binary tree. This means that one single decision tree is highly sensitive to small change of the **training data**. This is NOT robust.   

We usually use the **bagged decision tree**, which is the collection of the decision trees, to vote for the prediction and we choose the majority as the final result. In this way, small change of the **training data** can impact little on our prediction.   

We assume that the number of the **training examples** is m and we would like to train the **bagged decision tree** of which the number of the decision trees is B.  

First, we use the **sampling with replacement** to create B new **training data sets**, of which the size is the same m, based on the original **training data set**. And then we train each single decision tree on each new **training data set**, and we will totally have B decision trees.  

We aim to create new **training data sets** which are a bit similar to but also pretty different from the original **training data set**. In this way, we can have more accurate prediction.  

The **sampling with replacement** means that, when picking the next one, first replacing all previously drawn ones into the set we are picking from. This means that the original **training example** may be repeated in our new **training data sets**.  

The reason to use the **sampling with replacement** is to make almost all the B trained decision trees consequently choose the same feature for the root node.  

The recommended value for B should be between 64 and 128 (for example, 100). Too large B value can cause diminishing returns.  

### Random Forest  

We assume that we have n **features**.  

We do NOT choose the **feature** with the highest **information gain** from these n **features**. Instead, we first randomly  select k **features** from these n **features**, and then we choose from these k **features**. The typical value of k is $\displaystyle k = \sqrt{n}$.  

In this way, the **bagged decision tree** is converted to the **Random Forest**.  

### Boost Decision Tree

We still use **sampling with replacement** to create B new **training data set**, but we do NOT sample from all m **training examples** with equal probability $\displaystyle \frac{1}{m}$. Instead, when we are creating the b-th **training data set**, we first collect all the previous b - 1 decision trees together as the **bagged decision tree** to conduct the prediction, and then we assign higher probabilities to the misclassified **training examples** when sampling.  

In this way, the **bagged decision tree** is converted to the **boost decision tree**. We focus more on the **training examples** which we are NOT yet doing well on.  
  
The **XGBoost** (**eXtreme Gradient Boosting**) is the popular implementation of the **boost decision tree**.  

Technically, XGBoost does NOT actually use the **sampling with replacement** or **random feature choice** but some more efficient methods instead. But we can still use the idea of **sampling with replacement** or **random feature choice** to comprehend the XGBoost.  

```python
import xgboost  

# based on the type of y to choose classifier or regressor  
model = xgboost.XGBClassifier() 
# model = xgboost.XGBRegressor()  


mode.fit(X_train, y_train)  

y_pred = model.predict(X_test)  
```

## Neural Network  

### Decision Tree

The **decision tree** works well on tabular (structured) **training data**(for example, the data stored in the spreadsheet), but does NOT work well on unstructured **training data**(for example, image, audio, text).  

It is usually faster to train the **decision tree** than the **neural network**.  

The small decision tree is even human interpretable.  

### Neural Network  

The **neural network** works well on all types of **training data** (for example, image, audio, text).

It may be slower to train the the **neural network** than the **decision tree**. But we can use the **transfer learning** to train the model on the small **training data** based on the **coefficients** of the **hidden layers** of the existing pretrained model.  

When we would like to make multiple models work together, it is usually easier to train the muiltiple **neural networks** than the multiple **decision trees**.  
