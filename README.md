# Andrew's Netflix playground 

This is a repo for me to mess with Recommender Engines, with the Netflix prize data as a playground.
I used the repo at alexbw as a starting point, mainly for his functions to read netflix data into numpy arrays, however the Kfolds sampling and all the algorithms are rewritten.

The plan is to implement the "recipes" summarized in Recommender Systems (Aggarwal, 2016).

Particularly, I am interested in the performance of: Collaborative filtering methods (user-based) Latent factor methods

## Packages used
* cython 0.27.3
* numpy: 1.13.3
* scipy: 1.0.0 (for sparse matrix representations and conversions between)
* scikit-learn: 0.19.1
