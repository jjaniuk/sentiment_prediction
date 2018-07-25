## Sentiment Prediction - Amazon Reviews

## Background

The goal will be to predict whether a review is positive or negative based on the comments left by the user. We will also look at the usefulness of a comment to other users based on the feedback from other users.

## Data Exploration and transformation

From some data exploration we see that the dataset has 173 000 observations. 
Some things to note:
1. There is a skew towards 5 star ratings 
2. We can remove the 3 star ratings, as these would classify as neutral ratings
3. We create a "sentiment" observation, which determines if a rating is "positive" (5 or 4 stars), or "negative" (2 or 1 star)
4. We create a "usefulness" observation, which determines if a user comment is useful (useful if 80% of users found it helpful, and uselss if less than 80% of users found it helpful)
5. We see that the data is largely skewed towards the positive (148 657 positive ratings vs. 24 343 negative ratings)
6. Due to point #5 we need to re-sample the dataset to remove the skewness.
7. We use the wordcloud module to visualize the most comment words to get a feeling for the different words we will be working with.

## Feature Engineering

We use the CountVectorizer() and TfidfTransformer() to build our features. This will take one to four words and create a coefficient for the most commonly used words. A positive float coefficient will mean it is a positive rating and a nagative float coefficient will be a negative rating to put into simple terms.

## Modeling

### Positive/Nagative Rating Prediction

We will try to apply three different models: [Multinomial Naïve Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html), [Bernoulli Naïve Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html), [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) and compare them using the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

Based on the ROC curve, we see that the logistic regression model is best suited for our dataset, as it has the best precision and recall out of the three. On average, this model is able to predict whether a review is positive or negative at a 91% accuracy.

### Usefulness of Comment Prediction

Again, we have to re-sample the data, as it is skewed towards the useless comments (most comments were not helpful for other users).

We re-build our features and run the logistic regression model. In this case, we have much lower prediction %, at about 61%. This might mean a few things.

1. Comments which are useful for most users don't have much difference in those that are not useful/have no votes.
2. We need to work more on refining and tweaking the features in order to predict the difference in comments that are useful and useless for this group of users.