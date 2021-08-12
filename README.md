# BiasedAbstractiveSummarizer
Biased Abstractive Summarizer 

This is a combination of a sentiment analysis LSTM and an abstractive summarizer using the sentiment analysis LSTM parameters to summarize text. NOTE that the abstractive summarizer is not trained in this model since the training parameter file is too large for github, so while the sentiment analysis will work out of the box, the abstractive summarizer will require training. 

# Sentiment LSTM
To run sentiment analysis front-end, download the repo and run sentiment/main.py. It should lead to a prompt for you to input your text and determine its sentiment. The data is already preloaded and you can retrain the algorithm. You may need to install some modules in order for it to work, but it will give warnings letting you know which modules to install. You may also need to rename some filepaths, though this shouldn't be the case. 

# Abstractive Summarizer 
An example training dataset can be found here: https://www.kaggle.com/snap/amazon-fine-food-reviews

Download the dataset and store it as summary/data/reviews.csv. Run summary/main.py. It will ask you for permission to clean the data before it begins training. To alter the parameters of the training set to go faster/slower, adjust the variable DIM in main.py. To see sample sentiment/summaries, uncomment lines 281-285. NOTE that some training datasets may be mislabeled by pandas (known issue) so ymmv. 



