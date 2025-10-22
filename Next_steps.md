## Next steps

Next steps here is to find a more robust way to generate the data and evaluate similarity
One possible way is to generate similar sentences using transformers and use imitation learning, ie 
how does the model's similarity score compare to the transformers similarity score. 
Another would be to use k-mean or other clustering techiniques (supervised or unsupervised), only one
word in a sentence is varied and the inter-cluster distance is maximised for dissimilar words and where as
similar word should be found near the center of the cluster