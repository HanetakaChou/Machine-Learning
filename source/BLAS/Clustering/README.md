# K Means  

 // unsupervised learning // unlabeled data  


$\displaystyle \mu_1 = \frac{1}{4} \left[ x^{(1)} + x^{(5)} + x^{(6)} + x^{(10)} \right]$  // x^{(1)} means the first training example  // each value is n vector (n features)  

// sometimes 0 points // more common: eliminate the cluster k = k - 1 // also possible to initial at random position  

// works well even if the data is not well seperated  

// cluster centroid provide the sense of the most representive 

// is intrisically to optimize the WCSS(within-cluster sum of squares) cost funtion

C (cluster assignments): length m is the same as the examples  

// c^{(i)}  // index of cluster to which example x(i) is currently assigned 

/mu (centroids) : length k is the same as clusters  

// /mu_k  cluster centroid of k-th cluster  // n vector (same as features)  


// /mu_{c^{(i)}} cluster centroid of the cluster to which example x(i) is currently assigned


WCSS (within-cluster sum of squares) cost function J(C, /mu) // of which length of c is m and length of /mu is k  
// also called distortion function  

$\displaystyle \frac{1}{m} \sum_{i=1}^m {\| x^{(i)} - \mu_{c^{(i)}} \|}^2$

coordinate descent  

assignment step:  
assign each point to the nearest centroid (minimizing WCSS with respect to assignments) 
centroids are fixed, and the asiignments (cluster labels) are optimized  

update step:
recalculate centroids as the mean of the points in each cluster (minimizing WCSS with respect to centroids)  
asiignments are fixed, and the centroids are optimized

// keep mu fixed, optimize c

$\displaystyle c^{(i)} = \arg \min \limits_{k} {\| x^{(i)} - \mu_k \|}^2$  // can be 1, 2, 3 ... k

move cluster centroids 

// keep c fixed, optimize mu

$\displaystyle \mu_k = \bar{x}_k$ where $\displaystyle x_k = \left\{ x^{(i)} \mid c^{(i)} = k \right\}$  

// the cost function can never go up // otherwise, there is bug // unlike gradient descent  

// cost will either decrease or stay the same after each iteration  

// why can converge  

// cost update less than threhold //like the gradicent descent  

// or cost remain the same // no change per iteration  

// random intialization  
// initial for \mu_k 

// usually K < m // we can randomly picking K examples  

// non-convex // different initialization can lead to different results // local minima  

// one way // run k-means multiple times (common choice 50 - 1000) and choose the lowest cost function  

// If two centroids start too close to each other, they might end up capturing a single cluster and "competing" for it, leaving the third cluster poorly represented  

// The algorithm might stop at a point where moving the centroids further apart would temporarily increase the cost, even though it could lead to a better overall solution // cost can never go up // unlike gradient decsent  

// choose number of K  



// one way is to use Elbow method // but the author do not use it by himself (not recommend) // the author claims that for many applications the K is ambiguous   

// check the relatioship between cost function and K // find the elbow // the cost function decreases rapidly before the elbow and decreases slowly after the elbow // sometimes, there is no clear elbow  

// note: choose K to minimize the cost is not correct // this will always end up the large k  

// choose the number of k based on the later(downstream) purpose   

// T shirt size // k=3 S M L // k=5 XS S M L XL  
// trade-off

// image compressing // trade-off between quality and performance  







