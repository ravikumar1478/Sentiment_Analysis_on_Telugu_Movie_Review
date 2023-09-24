# Sentiment_Analysis_on_Telugu_Movie_Review
---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: Telugu_movie_review_sentiment
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Telugu_movie_review_sentiment

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3627
- Accuracy: 0.8814
- F1: 0.8889

## Model description

it is also a NLP related project that too dealing with Telugu dataset.
it is very useful project for the people who only known's Telugu Language.
we developed a sentiment analysis language model with rating generator, 
we Created a rating generator that gives a rating for a movie review based on the model output probabilities. According to the positive or negative polarity probabilities obtained from the review, we can generate a rating for that movie. my role in this project is creation of dataset, we take one sentence about the movie and manually given the label such as Positive and Negative
so that during the training the model will capture the polarity words which can understand machine which statement is negative and positive.
and another work is creating in documents 

## Intended uses & limitations

### Intended Uses:
Sentiment Analysis for Telugu Movie Reviews: This project is designed for Telugu-speaking users, allowing them to analyze the sentiment of Telugu movie reviews.
Movie Rating Generator: It provides a quick movie rating based on sentiment analysis results, assisting Telugu-speaking audiences in decision-making.
Training Data Creation: This project offers a manually labeled dataset for Telugu movie reviews, aiding NLP research in the field of sentiment analysis.
Documentation: Comprehensive documentation is available to guide users and developers.

### Limitations:
Language Restriction: Only supports the Telugu language.
Sentiment Accuracy: Accuracy depends on the input text's quality and complexity.
Dataset Bias: Accuracy is influenced by biases in the training data.
Movie-Specific Content: May not handle reviews with obscure references well.
Model Updates: Requires regular updates to maintain accuracy.
External Factors: Accuracy may be affected by user bias and industry changes.

## Training procedure

### Step-1:
For the dataset we created our own dataset file which contains movie sentences(reviews) and its corresponding review that is "Positive" and "Negative"
we taken reference of The corpus "Sentiraama" was created by G.Rama Rohit Reddy at Language Technologies Research Centre, KCIS, IIIT Hyderabad,In the corpus, folder named "Movie Reviews" contains 267 different Telugu movie reviews written in Telugu script. Out of them 136 are positive and 131 are negative. It contains a total of 20000 sentences and 165049 words. In the corpus, folder named "Product Reviews" contains 200 different product reviews written in Telugu script. Out of them 100 are positive and 100 are negative. It contains a total of 43199 sentences and 259189 words, these sentences are very long(consisting of whole story of the movie) so decided to make it short for easy training .

### Step2:
Pre-processing Data The collected data is pre-processed using different pre-processing techniques and splitting the large Telugu Sentence into small sentences.

### Step-3:
Connecting to Hugging Face Hugging Face provides a token with which we can log in using a notebook function and the rest of the work we do will be exported to the platform automatically.

### Step-4: 
Loading pre-trained model and tokenizer The pre-trained model and tokenizer from xlm-roberta-base are loaded for training our Telugu data

### Step-5: 
Training the model Required libraries like Trainer and Training arguments are imported from Transformers library. The after giving the Training arguments with our data we train the model using the train() method which takes 1 to 1 ½ hours depending upon the size of our input data

### Step-6: 
Pushing model and tokenizer Then trainer.push_to_hub() and tokenizer.push_to_hub() methods are used to export our trained model and its tokenizers which are used for the mapping of words in prediction.

### Step-7: 
Testing In the hugging face after opening our model page there is an API in which We give a Telugu Sentence as input with

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4
- 
### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu118
- Datasets 2.11.0
- Tokenizers 0.13.3
