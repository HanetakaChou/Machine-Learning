# Logistic Regression

## Sigmoid Function

We have the **training data** of which y is the **target** (or **output**) label of the **binary classification**, and X is n **features**  (or **inputs**).  

We assume that the **probability** of the **class 1** can be modeled by the **sigmoid function** $\displaystyle \mathop{\mathrm{P}}(y = 1) = \hat{y} = \mathop{\mathrm{\sigma}}(z) = \mathop{\mathrm{\sigma}}(X \theta) = \frac{1}{1 + \exp(- (X \theta))}$ of which $\displaystyle \hat{y}$ is the **prediction** which is estimated by our model and should be distinguished from the original **target** y, and $\displaystyle \theta$ is n **weights** which we would like to estimate.  

We have m **training examples**. This means that the dimension of y is m × 1, and the dimension of X is m × n. 

We would like to estimate $\displaystyle \theta$ of which the demension is n x 1.  

## Linear Hyperplane Decision Boundary  

For the **binary classification**, we can learn from the **sigmoid function** that $\displaystyle \begin{cases} \mathop{\mathrm{\sigma}}(X \theta) \ge 0.5 & X \theta \ge 0 \\ \mathop{\mathrm{\sigma}}(X \theta) < 0.5 & X \theta < 0 \end{cases}$. (Technically, the threshold may NOT be 0.5, for example, when the **skewed dataset** is involved.)  
  
This means that the **linear hyperplane** $\displaystyle X \theta = 0$ divides these m **training samples** into 2 **classes**. These **training samples** above the **linear hyperplane** follow $\displaystyle X \theta \ge 0$ and belong to the **class 1**, while these **training samples** below the **linear hyperplane** follow $\displaystyle X \theta < 0$ and belong to the **class 0**.  

This **linear hyperplane** $\displaystyle X \theta = 0$ is also called the **decision boundary**. The aim of the **logistic regression** is intrisically to find the **decision boundary** which can divide these m **training samples** into 2 **classes**.  

## Intercept Term 

We need to add one more constant feature column (of which the value remains 1) to calculate the **intercept** (or **bias**) term of the **linear hyperplane**, just as what we do for the **linear regression**.  

## Feature Scaling  

We need to use the **feature scaling** to ensure the **numerical stable**, just as what we do for the **linear regression**.  

## Feature Transformation  

We can use the **feature transformation** to implement the **non-linear decision boundary**, just as what we do for the **linear regression**.  

For example, we have the **circular decision boundary** $\theta_1 + \theta_2 x_1^2 + \theta_3 x_2^2 = 0$. We only need to consider the features $\displaystyle \begin{bmatrix} 1 & x_1^2 & x_2^2 \end{bmatrix}$, and the problem has been converted to the **linear decision boundary**.  

## MLE (Maximum Likelihood Estimation)  

For the **binary classification**, we have $\displaystyle \mathop{\mathrm{P}}(y = 0) + \mathop{\mathrm{P}}(y = 1) = 1$, and we have the **probability** of one single **training example** $\displaystyle \mathop{\mathrm{P^{(i)}}}(\theta) = \mathop{\mathrm{P^{(i)}}} (y = y^{(i)}) = \begin{cases} \hat{y} & y^{(i)} = 1 \\ 1 - \hat{y} & y^{(i)} = 0 \end{cases} = {\hat{y}}^{y^{(i)}} \cdot {(1 - \hat{y})}^{1 - y^{(i)}}$.  

We assume these m **training examples** are **independent**, and we have the **log-likelihood function** $\displaystyle \ln \mathop{\mathcal{L}}(\theta) = \ln \left\lparen \prod_{i=1}^m \mathop{\mathrm{P^{(i)}}}(\theta) \right\rparen = \ln \left\lparen \prod_{i=1}^m {\hat{y}}^{y^{(i)}} \cdot {(1 - \hat{y})}^{1 - y^{(i)}} \right\rparen = - m \left\lparen - \frac{1}{m} \sum_{i=1}^m \left\lparen y^{(i)} \ln (\hat{y}) + (1 -  y^{(i)}) \ln(1 - \hat{y}) \right\rparen \right\rparen$.  

The $\displaystyle \mathop{\mathrm{H}}(\theta) = - \frac{1}{m} \sum_{i=1}^m \left\lparen y^{(i)} \ln (\hat{y}) + (1 -  y^{(i)}) \ln(1 - \hat{y}) \right\rparen$ term is also called the **cross-entropy loss function**. We need to **minimize** the **cross-entropy loss function** to **maximize** the **log-likelihood function**. The problem has been converted to find the **minimize** of the **cross-entropy loss function**.   

## Cost Function  

We select the **cross-entropy loss function** as the **cost function** $\displaystyle \mathop{\mathrm{J}}(\theta) = \mathop{\mathrm{H}}(\theta) = - \frac{1}{m} \sum_i^m \left\lparen y^{(i)} \ln (\hat{y}) + (1 -  y^{(i)}) \ln(1 - \hat{y}) \right\rparen$, and we have the **gradient** $\displaystyle \nabla \mathop{\mathrm{J}}(\theta) = \frac{\partial \mathop{\mathrm{J}}(\theta)}{\partial \theta} = \frac{1}{m} \sum_{i=1}^m \left\lparen X^{(i)} ({\hat{y}}^{(i)} - y^{(i)}) \right\rparen = \frac{1}{m} X^T (\mathop{\mathrm{\sigma}}(X \theta)- y) = \frac{1}{m} X^T (\mathop{\mathrm{\sigma}}(z)- y) = \frac{1}{m} X^T (\hat{y} - y) = \frac{1}{m} X^T \epsilon$.  

## Convex Optimization  

Fortunately, the **cross-entropy loss function** is **convex**.  

## L2 Regularization (Ridge Regularization)  

We can add the same **regularization term** $\displaystyle \frac{\lambda}{2 m} \sum_{j=2}^n \theta_j^2$ to the **cost function** to avoid the **overfitting**, just as what we do for the **linear regression**.  

## Multi-label Classificaton  

The **multi-label classification** should be distinguished from the **multiclass classification**.   

For the **multiclass classification**, the target is the **one-hot** vector, and we use the **softmax function** to predict the possibility of the whole vector.  

However, for the **multi-label classification**, the target is still the vector but NOT no longer **one-hot**, and we use the **sigmoid function** to predict the possibility of each label of the vector.  
