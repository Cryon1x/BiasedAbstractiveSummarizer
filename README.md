# BiasedAbstractiveSummarizer
Biased Abstractive Summarizer 

This is a combination of a sentiment analysis LSTM and an abstractive summarizer using the sentiment analysis LSTM parameters to summarize text. NOTE that the abstractive summarizer is not trained in this model since the training parameter file is too large for github, so while the sentiment analysis will work out of the box, the abstractive summarizer will require training. 

# Sentiment LSTM
An example training dataset can be found here: https://www.kaggle.com/c/tweet-sentiment-extraction/overview. 

To run sentiment analysis front-end, download the repo and run sentiment/main.py. It should lead to a prompt for you to input your text and determine its sentiment. The data is already preloaded so you can retrain the algorithm without needing to download the tweet-sentiment-extraction data. You may need to install some modules in order for it to work, but it will give warnings letting you know which modules to install. You may also need to rename some filepaths, though this shouldn't be the case. 

# Abstractive Summarizer 
An example training dataset can be found here: https://www.kaggle.com/snap/amazon-fine-food-reviews

Download the dataset and store it as summarylib/data/reviews.csv. Run summary/lib/main.py. It will ask you for permission to clean the data before it begins training. To alter the parameters of the training set to go faster/slower, adjust the variable DIM in main.py. To see sample sentiment/summaries, uncomment lines 281-285. NOTE that some training datasets may be mislabeled by pandas (known issue) so ymmv. Also note that an attention mechanism was not implemented in keras by default, so we  used code from https://github.com/thushv89/attention_keras for our attention layer implementation. The file is called "borrowed_attention.py". 



