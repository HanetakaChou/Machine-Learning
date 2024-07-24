# Linear Regression  

## Linear Model

We have the **training data** of which y is the **target** (or **output**), and X is n **features**  (or **inputs**).  

We assume the **linear model** $\displaystyle \hat{y} = \mathop{\mathrm{f_{\theta}}}(X) = X \theta$ of which $\displaystyle \hat{y}$ is the **prediction** which is estimated by our model and should be distinguished from the original **target** y, and $\displaystyle \theta$ is n **weights** which we would like to estimate.  

We have the **error** $\displaystyle \epsilon = \hat{y} - y = \mathop{\mathrm{f_{\theta}}}(X)- y = X \theta - y$ which should ideally be **minimized**.  

We have m **training examples**. This means that the dimension of y is m × 1, and the dimension of X is m × n.  

We would like to estimate $\displaystyle \theta$ of which the demension is n x 1.  

## Intercept Term  

Usually, the **intercept** (or **bias**) term  is considered as the part of the X. We add one more constant feature column (of which the value remains 1) to implement that.  

The **linear regression**, which only 1 feature is involved, is called the **univariate** linear regression. But the n should be 2 instead of 1, since one more constant feature column is added to estimate the **intercept term**.  

The **weights** and the **bias** are collectively called the **coeffiects**. In this case, we have totally n + 1 **coeffiects**.  


## MLE (Maximum Likelihood Estimation)  

We assume that the error $\displaystyle \epsilon$ follows the **normal distribution** $\displaystyle \mathop{\mathrm{N}}(0, \sigma)$. This means that $\displaystyle \mathop{\mathrm{p}}(\epsilon) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\lparen - \frac{\epsilon^2}{2 \sigma^2} \right\rparen$  

We assume that these m **training examples** are **independent**, and we have the **log-likelihood function** $\displaystyle \ln \mathop{\mathcal{L}}(\theta) = \ln \left\lparen \prod_i^m \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\lparen - \frac{{(y^{(i)} - X^{(i)}\theta)}^2}{2 \sigma^2} \right\rparen \right\rparen = \sum_i^m \left\lparen -\frac{1}{2} \ln \left\lparen \sqrt{2 \pi \sigma^2} \right\rparen - \frac{{\left\lparen y^{(i)} - X^{(i)}\theta \right\rparen}^2}{2 \sigma^2} \right\rparen = -\frac{m}{2} \ln \left\lparen \sqrt{2 \pi \sigma^2} \right\rparen - \frac{1}{2 \sigma^2} \sum_i^m {\left\lparen y^{(i)} - X^{(i)}\theta \right\rparen}^2$.  

The $\displaystyle \sum_i^m {\left\lparen y^{(i)} - X^{(i)}\theta \right\rparen}^2$ term is also called the **SSE (Sum of Squared Errors)**. We need to **minimize** the **SSE (Sum of Squared Errors)** to **maximize** the **log-likelihood function**. The problem has been converted to find the **minimum** of the **SSE (Sum of Squared Errors)**.   

## Cost Function  

We select the half of the **MSE (Mean Squared Error)** as the **square error cost function** $\displaystyle \mathop{\mathrm{J}}(\theta) = \frac{1}{2} \mathop{\mathrm{MSE}}(\theta) = \frac{1}{2 m} {(y - X\theta)}^T(y - X\theta)$, and we can calculate the **gradient** more conveniently, since the "2" can be reduced $\displaystyle \nabla \mathop{\mathrm{J}}(\theta) = \frac{\partial \mathop{\mathrm{J}}(\theta)}{\partial \theta} = \frac{1}{m} ( {X^T} X \theta - {X^T} y)$. The problem has been converted to find the **minimum** of the **square error cost function**.  

## Normal Equation (Analytical)  

The **normal equation** is based on the **OLS (Ordinary Least Squares)**.  

We try to find the **zero** of the **gradient** to find the **minimum**.   

For **linear regression**, this means that we have the linear equation $\displaystyle \nabla \mathop{\mathrm{J}}(\theta) = \frac{1}{m} ( {X^T} X \theta - {X^T} y) = 0$. And we can also have the analytical solution, which is also called the **normal equation**, for this linear equation $\displaystyle \theta = {({X^T} X)}^{-1} {X^T} y$.  

## Gradient Descent (Numerical)  

The **gradient descent** is intrinsically to find the **minimum** of $\displaystyle \mathop{\mathrm{J}}(\theta)$ based on **iteration**.  

We have $\displaystyle {\theta}^{(t+1)} = {\theta}^{(t)} - \alpha \nabla \mathop{\mathrm{J}}({\theta}^{(t)})$ of which the $\displaystyle \alpha$ is the **learning rate** which should be carefully tuned.  

And the **gradient** can also be rewritten as $\displaystyle \nabla {\mathrm{J}}(\theta) = \frac{1}{m} ( {X^T} X \theta - {X^T} y) = \frac{1}{m} {X^T} (X \theta - y) = \frac{1}{m} {X^T} (\hat{y} - y) = \frac{1}{m} {X^T} \epsilon$ to reduce the amount of calculation.  

Usually, we record the value of $\displaystyle \mathop{\mathrm{J}}(\theta)$ after each iteration as the **cost history** to check **convergence**. If the value of $\displaystyle \mathop{\mathrm{J}}(\theta)$ increases after one iteration, the **learning rate** may be too high. If the decreasing of the value of $\displaystyle \mathop{\mathrm{J}}(\theta)$ is less than the **cost change threshold** $\displaystyle \gamma$, the **gradient descent** can be considered to have **converged**.  

## First-Order Optimization

For the **normal equation**, we try to find the **zero** of the **gradient** to find the **minimum**.  

In contrast, for the **gradient descent**, the **zero** of the **gradient** is not involved. This makes the **gradient descent** easy to implement, but the annoying **learning rate** $\displaystyle \alpha$, which should be carefully tuned, is introduced.  

However, there is also other **iteration** methods which can be used to find the **zero** of the **gradient**. For example, we all know the **Newton's Method** from the calculus class. The **Newton's method** can be used to find the **zero** of the **gradient**  $\displaystyle {\theta}^{(t+1)} = {\theta}^{(t)} - H^{-1} \nabla \mathop{\mathrm{J}}({\theta}^{(t)})$ and the **learning rate** $\displaystyle \alpha$ can be eliminated. But the gradient of the gradient, namely **Hessian**, is introduced. This makes the **Newton's method** more complex to implement.  

Actually, the **gradient descent** is intrinsically the **first-order optimization**, while the **Newton's method** is intrinsically the **second-order optimization**.  

Although there are some other improved variants of the **gradient descent**, such as **Adam (Adaptive Moment Estimation)**,  but the **learning rate** $\displaystyle \alpha$ is still required.  

## Convex Optimization

Both the **normal equation** and the **Newton's method** try to find the **zero** of the **gradient** to find the **minimum**.  

But it should be noted that the position, where the **gradient** is **zero**, is NOT necessarily the **minimum**, unless the function is **convex**.  

Actually, all the **normal equation**, the **gradient descent** and the **Newton's method** do NOT work for the **non-convex** function.  

## Feature Scaling  

The **feature scaling** reduces the **condition number** of the matrix. The higher condition number indicates the matrix is close to singular (non invertible). This is the reason why the **normal equation** can be benefited from the **feature scaling**.  

At the same time, the **feature sacling** makes the **gradient** more evenly distributed in different directions. We can learn from the **contour plot** by intuision that the **gradient** is more **numerical stable** in this case. And thus, the **gradient descent** can also be benifited from the **feature scaling**.    

## Feature Transformation  

The **feature transformation** is intrinsically to convert the features as the form of the **linear combination**.  

For example, we have the **polynomial model** $\displaystyle y = \theta_1 + \theta_2 x + \theta_3 x^2$. We only need to consider the whole $\displaystyle x^2$ as a new feature. This means we have 2 features $\displaystyle \begin{bmatrix} 1 & x & x^2 \end{bmatrix}$ (of which the 1 is the intercept term), and the problem has been converted to the **linear regresssion**.  

Actually, the model can be arbitrary, as long as the features can be converted as the form of the **linear combination**. For example, we can have the model $\displaystyle y = \theta_1 + \theta_2 x + \theta_3 \sin x$. We only need to consider the features $\displaystyle \begin{bmatrix} 1 & x & \sin x \end{bmatrix}$, and the problem can also be converted to the **linear regresssion**.  

We only assume that these m **training examples** are **independent** in the MLE (Maximum Likelihood Estimation). But these n **features** can be **dependent**.  

## L2 Regularization (Ridge Regularization)  

A model is considered **underfitting** (or **high bias**) if the model is too simple and work poorly even on the training data.  

A model is considered **overfitting** (or **high variance**) if the model is too complex and fits the **noise** of in the training data, which makes the model work well on the training data but work poorly on the new unseen data, namely poor **generalization**.  

A model is considered **good fit** if the model works well on the training data without the **noise**.  

We can use **feature selection** to select a subset of the features and discard the rest to avoid the **overfitting**.  

We can also use the **L2 regularization (Ridge regularization)** to avoid the **overfitting**.  

The **regularization term** $\displaystyle \frac{\lambda}{2 m} \sum_{j=2}^n \theta_j^2$, where the $\displaystyle \lambda$ is the **regularization parameter** and the **intercept term** $\displaystyle \theta_1$ is usually excluded, is added to the **cost funtion** to penalize the coefficients.  

The derivative term $\displaystyle \frac{\lambda}{m} \begin{bmatrix} 0 & \theta_2 & \cdots & \theta_n \end{bmatrix}$ is added to the **gradient**.  

Actually, the **gradient descent** can also be rewritten as $\displaystyle \theta^{(t+1)} = (1 - \alpha \frac{\lambda}{m}) \theta^{(t)} - \alpha \nabla \mathop{\mathrm{J}}({\theta}^{(t)})$  (where the **intercept term** $\displaystyle \theta_1$ should technically be excluded). This means that the **L2 regularization (Ridge regularization)** is intrinsically to shrink the coefficients by a **fixed** rate $\displaystyle 1 - \alpha \frac{\lambda}{m}$ for each iteration.  
