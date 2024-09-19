Content-based filtering:  
Recommend items to you based on features of user and item to find good match  // no need the data from other users  

predict user vector from user input feature vector x_u   

predict item vector from item input feature vector x_v  

the dim of user features and item features varies a lot  

but the dim of user vector and item vector should be the same  

the meaning of the user vector and item vector  
// represent the latent factor  
// e.g. user vector // likes action // likes romance  
// item vector // action // romance  

dot(V_u, V_m) -> regression  
sigmoid(dot(V_u, V_m)) -> classification  

// find another item j similiar to item i  
// calculate the distance between the latent factor vector {\| V_m_j - V_m_i \|}^2  
// can be precomputed a head of time // after the precomputing, then recommend to the users  

// the item vector can be precomputed before we know x_u v_u (of the user)  
 
large catalogue  
how to efficiently find recommendation from a large set of items  
// we do not predict for each item  

Two steps: Retrieval and Ranking  

Retrieval  
Generate large list of plausible item candidates // tries to cover a lot of possible items you might recommend to the user // during this step, it is fine to include lots of items which the user is not likely to like  
e.g. for each of the last 10 movies watched by the user, find 10 most similar movies  
for most viewed 3 genres, find the top 10 movies  
top 20 movies in the country  
Combine retreived items into list, removing duplicates and items already watched purchased  

// retrieving more items results in same or higher quality, but slower recommendations  
// to analyse/optimiza the trade-off, carry out offline experiments  
// to see if retrieving additional items results in more relevant recommendations (i.e. p = 1 of the items displayed to user are higher)  

Ranking  
take the list retreved and rank using learned model  
display ranked items to user  

// the user vector v_u can be precomputed for each user  

// basic content based filtering:  

// item representation vector v:   

// user profile vector u:  

// cosine Similarity  

// the item representation vector and user profile vector may be the learned representations // try to extract the low-dimensional learned  representations from the high-dimensional content-based input features  





