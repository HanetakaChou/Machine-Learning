
kernal density estimation  

translate K to xi  
if there are many same xi, K at xi will be enhanced (due to the sum)

gaussin (normal distribution) // Maximum likelihood estimation in 

z - score (feature normalization)
// use gussian distribution to predict the probability // multiple features // product them together (assume the features are indepenpend but it works well even if some of the features are not independent)  

amomalous when the probability lower than preset threshold // happen when one or more features are either too big or too small  

(real-number) evaluation  

1 amomalous 0 non-amomalous (normal): -> become labeled data (not for training set since we are still using unsurpervised learning)  
// skewed data ?

cross validation set and test set should include some amomalous examples 
but ideally no amomalous in training set (still unsurpervised learning // no label in training set) // even if we have the label, we can still use the unsurpvised learning  

we use the cross validation error to choose eplsion (threshold) 
// range epsilon from the min and max of gussian, calculate F1 score and choose the best (highest) one  


check test error 

// no alternsative: no test set // no enough data for amomalous examples to put in test set   

when to use anomaly detection
very small number of positive examples; large number of negative examples  
many different "types" of anomalies; hard for any algorithm to learn from anomalous examples what anomalies look like; **future anomalies may look nothing like any of the anomalous examples we've seen so far** // no anomalous examples in the training set  // for example (finacial fraud)  // (for manufacturing) find new previously **unseen** defects in manufacturing // monitoring (malfunctioning) computers a data center (security related software to find brand new ways of the hackers)  

when to use supervided learning (logistics regression)  
large number of both positive and negative examples  
enough anomalous examples for the algorithm to get a sense of what positive examples are like, **future positive examples likely to be siliar to ones in the training set**  // for example spam email // (for manufacturing) finding known previously **seen** defects (like the scratches) // weather prediction (seen weather over and over) // (specific, previous seen before) diseases classification  

choose what features to use // more important for anomaly detection // for supervised learning, we have the label ; after we train the model, the model can ignore the unrelated features automatically  

non-gaussian features // the histogram distribution does not look like the gaussian (matplot.hist(X, bins=...) to plot) // transform the feature to make it look more like the guassian (use matplot.hist to plot again) // apply same transform for cross validation and test 
// actually for maximum likeihood regression, we assume gaussian as well // we may apply the same technology  


error analysis  
p(x) is large (than threhold epsilon) / comparable for both normal and anomalous examples  // try to think why we still think it is anomouly // try to find new features  

monitor computers in data center // e.g. use ratio between two features // the feature should be either very large or very small for anomalous examples