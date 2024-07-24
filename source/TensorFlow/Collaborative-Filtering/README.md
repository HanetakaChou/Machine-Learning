
collaborative-filtering  
from multiple users  
items reviewed by different users should overlapped (e.g. the case, where user can purchase at most 1 item, is not suitable)  

// to identify what books are "similiar" to each other (i.e. if a user likes a certain book, what are other books that they might also like)  

m // examples  
n_u // users  
n // features  // x^{(i)}  

// different coefficients for different users // w^{(j)} b^{(j)}

// learn for each user y^{i,j} (target) 
// m^{(j)} // ignore the examples which the user has not rated  

 // for each user j // for the j-th user // use the cost function same as the linear regression (can also add the regularizaiton) 

// to learn the coefficients

$\displaystyle \mathop{\mathrm{J}} ( w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i \mid \mathop{\mathrm{r}}(i,j) = 1} {\left( w^{(j)} x^{(i)} + b^{(j)} - y^{(i, j)} \right)}^2 + \frac{\lambda}{2} \sum_{k=1}^n {\left( w_k^{(j)} \right)}^2$ // NOTE: we do not use $\displaystyle \frac{1}{2 m^{(j)}}$ and $\displaystyle \frac{\lambda}{2 m^{(j)}}  

 // for all users  
 // $\displaystyle \mathop{\mathrm{J}} ( w^{(1)}, \dots, w^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}) = \sum_{j=1}^{n_u} \mathop{\mathrm{J}} ( w^{(j)}, b^{(j)})$ // summ all cost functions together  

// when the features are not known, but we know the parameters for the users (similiar to linear regression as well) // note the regularization term different 

// to learn the features
$\displaystyle \mathop{\mathrm{J}} ( x^{(i)}) = \frac{1}{2} \sum_{j \mid \mathop{\mathrm{r}}(i,j) = 1} {\left( w^{(j)} x^{(i)} + b^{(j)} - y^{(i, j)} \right)}^2 + \frac{\lambda}{2} \sum_{k=1}^n {\left( x_k^{(i)} \right)}^2$ // NOTE: we do not use $\displaystyle \frac{1}{2 m^{(j)}}$ and $\displaystyle \frac{\lambda}{2 m^{(j)}} 

// for all users  
// $\displaystyle \mathop{\mathrm{J}} ( x^{(1)}, \dots, x^{(m)}) = \sum_{i=1}^{m} \mathop{\mathrm{J}} ( x^{(i)} )$

// put togther (collaborative-filtering)  
$\displaystyle \mathop{\mathrm{J}} ( w^{(1)}, \dots, w^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}, x^{(1)}, \dots, x^{(n_u)}) = \mathop{\mathrm{J}} ( w^{(1)}, \dots, w^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}) + \mathop{\mathrm{J}} ( x^{(1)}, \dots, x^{(n_u)}) = \frac{1}{2} \sum_{(i,j) \mid \mathop{\mathrm{r}}(i,j) = 1} {\left( w^{(j)} x^{(i)} + b^{(j)} - y^{(i, j)} \right)}^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n {\left( w_k^{(j)} \right)}^2 + \frac{\lambda}{2} \sum_{i=1}^{m} \sum_{k=1}^n {\left( x_k^{(i)} \right)}^2$  

// user feature matrix (similiar to coefficients in linear regression)  // row: number of users // column: number of latent factors (the features)  
// item feature matrix (smiliar to features in linear regression) // row: number of items // column: number of latent factors (the features)  
// user-item interaction matrix R (sparsity: extremely sparse) // we use the Embedding layer of the tensorflow to implement the sparse matrix  

// Embedding layer: fundamentally look up table // map the same ID to the same vector  
// input is multiple ID // output is (batch_size, 1, embedding_dim) // use Flatten to convert to (batch_size, embedding_dim)  
// input_dim is NOT related to batch_size // number of the unique ID // start from 0 // the range is [0, input_dim - 1]


// \kappa // the set of observed (non-missing) user-item pairs  

$\displaystyle \mathop{\mathrm{J}}(U, V) = \frac{1}{2} \sum_{(i, j) \in \kappa} {\left(R_{ij} - U_i V_j^T \right)}^2 + \frac{\lambda}{2} {\| U \|}^2 + \frac{\lambda}{2} {\| V \|}^2$

// use squared error instead of mean squared error  
// in keras, we only support mean squared error (operate on batch size)  

// use gradient descent  

// when R is the binary label (e.g. like or dislike // 1 (engage) 0 (not engage) ? (item not shown)) 
// apply sigmoid function to the previous linear model (similiar to logistic regression) // cross entropy ?  


$\displaystyle \mathop{\mathrm{J}}(U, V) = \sum_{(i,j) \in \kappa} \left( R_{ij} \ln \left( \mathop{\mathrm{\sigma}} \left( R_{ij} - U_i V_j^T \right) \right) + \left( 1 -  R_{ij} \right) \ln \left(1 - \mathop{\mathrm{\sigma}} \left( R_{ij} - U_i V_j^T \right) \right) \right) + \frac{\lambda}{2} {\| U \|}^2 + \frac{\lambda}{2} {\| V \|}^2$  

// mean normalization  
// users who have not rated any movies (all ? for all movies) // will end up with all coefficients (w and b) of this user zero // and the prediction for the rating (w x + b) will also be all zero  

// normalize row -> new user  
// normalize column -> brand new movie // no one has rated yet  

// use the input X - /mu to train // for this user: the prediction for the rating (w x + b + /mu) and will be the /mu not zero  

// find related items  

// for example, for item i with features (latent factors) x_i  
// try to find another item j similiar to item i  
// use sum squared error // \sum {(x_j_k - x_i_k)}^2 // equilvelent to the distance between vector {\| x_k - x_i \|}^2 

// limitations of collaborative filtering  

// cold start problem:  
//  new item // few users have rated  
//  new users // have rated few items  

// can not use side information about items or users // for example, the age of user // the studio of the item  

// collaborative filtering:  
// recommend items to you based on the ratings of users who gave similar ratings as you  
