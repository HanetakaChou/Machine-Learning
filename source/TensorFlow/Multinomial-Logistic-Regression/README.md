# Softmax Regression (Multinomial Logistic Regression)

## Softmax Function

We have the **training data** of which y is the **target** (or **output**) **one-hot** 1 × K vector to represent the labels of the **K classes** of the **multiclass classification**, and X is n **features**  (or **inputs**).  

We assume that the **probability** of the **class k** ($\displaystyle k = 1 \dots K$) can be modeled by the **softmax function** $\displaystyle \mathop{\mathrm{P}}(y = k) = \hat{y}_k = \mathop{\mathrm{softmax}}(X)_k = \frac{\displaystyle \exp(z_k)}{\displaystyle \sum_{n=1}^K \exp(z_n)} = \frac{\displaystyle \exp(X \theta_k)}{\displaystyle \sum_{n=1}^K \exp(X \theta_n)}$ of which $\displaystyle \hat{y}$ is the **prediction** (1 × K vector) which is estimated by our model and is evidently different from the original **target** y, and $\displaystyle \theta_k$ ($\displaystyle k = 1 \dots K$) is n **weights** which we would like to estimate.  

We have m **training examples**. This means that the dimension of y is m × K, and the dimension of X is m × n. 

We would like to estimate $\displaystyle \theta_k$ ($\displaystyle k = 1 \dots K$) of which the demension is n x 1.  

## Linear Hyperplane Decision Boundary  

For the **multiclass classification**, we can learn from the **softmax function** that the probability of **class i** and **class j** should be the same when $\displaystyle X \theta_i = X \theta_j$.  
  
This means that the **linear hyperplane** $\displaystyle X \theta_i = X \theta_j$ divides these m **training samples** between **class i** and **class j**. These **training samples** above the **linear hyperplane** follow $\displaystyle X \theta_i \ge X \theta_j$ and belong to the **class i**, while these **training samples** below the **linear hyperplane** follow $\displaystyle X \theta_i < X \theta_j$ and belong to the **class j**.  

This **linear hyperplane** $\displaystyle X \theta_i = X \theta_j$ is also called the **decision boundary**. The aim of the **multinomial logistic regression** is intrisically to find all the $\displaystyle C_K^2 = \binom{K}{2} = \frac{K (K - 1)}{2}$ **decision boundaries** which can divide these m **training samples** into K **classes**.  

## Intercept Term 

We need to add one more constant feature column (of which the value remains 1) to calculate the **intercept** (or **bias**) term of the **linear hyperplane**, just as what we do for the **linear regression**.  

## Feature Scaling  

We need to use the **feature scaling** to ensure the **numerical stable**, just as what we do for the **linear regression**.  

## Feature Transformation  

We can use the **feature transformation** to implement the **non-linear decision boundary**, just as what we do for the **linear regression**.  

For example, we have the **circular decision boundary** $\theta_{k,1} + \theta_{k,2} x_1^2 + \theta_{k,3} x_2^2 = 0$. We only need to consider the features $\displaystyle \begin{bmatrix} 1 & x_1^2 & x_2^2 \end{bmatrix}$, and the problem has been converted to the **linear decision boundary**.  

## MLE (Maximum Likelihood Estimation)  

For the **multiclass classification**, we have the **probability** of the i-th **training example** $\displaystyle \mathop{\mathrm{P^{(i)}}}(\theta) = \mathop{\mathrm{P}}(y = y^{(i)}) = {\hat{y}}_{y^{(i)}} = \frac{\displaystyle \exp(z_{y^{(i)}}^{(i)})}{\displaystyle \sum_{n=1}^K \exp(z_n^{(i)})} = \frac{\displaystyle \exp(X^{(i)} \theta_{y^{(i)}})}{\displaystyle \sum_{n=1}^K \exp(X^{(i)} \theta_n)}$.  

We assume these m **training examples** are **independent**, and we have the **log-likelihood function** $\displaystyle \ln \mathop{\mathcal{L}}(\theta) = \ln \left\lparen \prod_{i=1}^m \mathop{\mathrm{P^{(i)}}}(\theta) \right\rparen = \ln \left\lparen \prod_{i=1}^m \frac{\displaystyle \exp(X^{(i)} \theta_{y^{(i)}})}{\displaystyle \sum_{n=1}^K \exp(X^{(i)} \theta_n)} \right\rparen = - m \left\lparen - \frac{1}{m} \sum_{i=1}^m \left\lparen X^{(i)} \theta_{y^{(i)}} - \ln \left\lparen \sum_{n=1}^K \exp \left\lparen X^{(i)} \theta_n \right\rparen \right\rparen \right\rparen \right\rparen$.  

The $\displaystyle \mathop{\mathrm{H}}(\theta) = - \frac{1}{m} \sum_{i=1}^m \left\lparen X^{(i)} \theta_{y^{(i)}} - \ln \left\lparen \sum_{n=1}^K \exp \left\lparen X^{(i)} \theta_n \right\rparen \right\rparen \right\rparen$ term is also called the **cross-entropy loss function**. We need to **minimize** the **cross-entropy loss function** to **maximize** the **log-likelihood function**. The problem has been converted to find the **minimize** of the **cross-entropy loss function**.   

## Cost Function  

We select the **cross-entropy loss function** as the **cost function** $\displaystyle \mathop{\mathrm{J}}(\theta) = \mathop{\mathrm{H}}(\theta) = - \frac{1}{m} \sum_{i=1}^m \left\lparen X^{(i)} \theta_{y^{(i)}} - \ln \left\lparen \sum_{n=1}^K \exp \left\lparen X^{(i)} \theta_n \right\rparen \right\rparen \right\rparen$, and we have the **gradient** $\displaystyle \nabla \mathop{\mathrm{J}}(\theta)_k = \frac{\partial \mathop{\mathrm{J}}(\theta)}{\partial \theta_k} = \frac{1}{m} \sum_{i=1}^m X^{(i)} \left\lparen \frac{\displaystyle \exp(X^{(i)} \theta_k)}{\displaystyle \sum_{n=1}^K \exp(X^{(i)} \theta_n)} - \left\lbrack y^{(i)} = k \right\rbrack \right\rparen = \frac{1}{m} \sum_{i=1}^m X^{(i)} \left\lparen \mathop{\mathrm{P}}(y = k) - \left\lbrack y^{(i)} = k \right\rbrack \right\rparen$.  

## Convex Optimization  

Fortunately, the **cross-entropy loss function** is **convex**.  

## L2 Regularization (Ridge Regularization)  

We can add the same **regularization term** $\displaystyle \frac{\lambda}{2 m} \sum_{j=2}^n \theta_{k,j}^2$ to the **cost function** to avoid the **overfitting**, just as what we do for the **linear regression**.  
