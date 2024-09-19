// reduce the number of the features  

// take the feature which varies a lot  
// meaning degree of variation  
// new axis (new coordinate) // one new feature to represent two existing features  

// reduce demention  
// e.g. From 3D to 2D  
// from 50 features to 2D (data visualize)  

// covariance  
// for the features X_i and X_j of the m training examples X  
// we have the covariance Cov(X_i, X_j) = \frac{1}{m-1} \sum (X_i - \mu_X_i) (X_j - \mu_X_j)  

// indicate the dependency between features X_i and X_j   

// covariance matrix \Sigma  
// assume n features: X_1, X_2, \dots , X_n  
// X_c = X - mean(X)  
// \Sigma = \frac{1}{n - 1} X_c^T X_c  

// eigen vector indicates that // the matrix can not change the direction of the eigen vector  
// the rotation axis of the rotation matrix should be one eigen vector and the corresponding eigen value is 1  

eigen decomposition  

// Eigenvectors represent directions in the feature space, and eigenvalues indicate how much variance (or "stretch") occurs along those directions.  

// choose axis // choose the projection with the largest variance // (first) pricipal component  
// find axit to retain variance  
 
// second pricipal component should be orthgnal to the first pricipal component   
// third should be orthognal to both first and second  

// was used for speeding up the training of a supervised learning model (support vector machine)  // no longer suitable for deep learning // actually we even use input encode (frequency encode) to get more input features  
