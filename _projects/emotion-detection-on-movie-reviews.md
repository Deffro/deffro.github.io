---
title: "Emotion Detection on Movie Reviews."
excerpt: ""
header:
  image: "images/projects/preprocessing.png"
  teaser: "images/teasers/preprocessing.png"
categories:
  - Data Processing
  - Feature Selection
  - Machine Learning  
sidebar:
  - title: "Role"
    image: http://placehold.it/350x250
    image_alt: "logo"
    text: "Designer, Front-End Developer"
  - title: "Responsibilities"
    text: "Reuters try PR stupid commenters should isn't a business model"
---


The object of this project is the **Emotion Analysis** of sentences that are comming from movie reviews, using **Machine Learning**. An attempt will be made to construct a
**classifier** capable of classyfying a sentence in one of the 6 basic categories of emotion which are **anger, disgust, fear, happiness, sadness, surprise**, and the category of **non-emotion**.

Things I did include:

- Created a new **dataset** using **inter-annotator** agreement
- Applied **feature selection** with **tf-idf** and **chi-square** methods
- Performed 15 **text pre-processing techniques** and their combinations
- Created and tested 14 new features in **feature engineering**
- Applied **ensemble learning**
- Used the resulting classifier in 6 movies

## 1. Creating the Dataset

The [IMDbPY](http://imdbpy.sourceforge.net/) was used, a free Python package for the retrieval and manipulation of data from the largest movie database [IMDb](http://www.imdb.com/).
Movies with more than 40 reviews were randomly retrieved and their reviews were stored in a file. Subsequently, representative **keywords** were chosen for each emotion category. These words can be seen in the following table.

| Emotion        | Keywords           |
| ------------- |:-------------:|
| Anger     | angry, annoyed, gloomy, depressed, unhappy, down, disheartened, sorrowful, painful, guilty |
| Disgust     | stupid, sucks, irritated, humiliated, disgusted, nauseatng, sickening, contempt, repelling, unpleasant      |
| Fear | afraid, frightened, fearful, horrified, nervous, paniched, alarmed, phobia, scared, insecure      |
| Happiness | awesome, happy, amused, fantastic, excited, pleased, cheerful, love, great, amazing      |
| Sadness | sad, lonely, gloomy, depressed, unhappy, down, disheartened, sorrowful, painful, guilty      |
| Surprise | astonished, bewildered, survrised, confused, sudden, unaware, shocked, perplexed, what, unexpected     |

Reviews were extracted that contained the above words, resulting in a total of **2,514** sentences. We need **labels** for **supervised** machine learning, so **3 human judjes** were selected to classify each sentence into one category.
The question that they had to answer was **"Classify the sentence according on the sentiment it exudes"**. The choice of the no-emotion and mixed-emotion were also possible. The judjes were first trained on already classified examples in order to be more confident.

It is very normal that the human judges will disagree on several sentences. There exist metrics which quantify the agreement between the judges called **inter-annotator agreement**.
I used **Cohen's kappa** which is used to compare the degree of consensus among reviewers in categorizing sentences into mutually exclusive categories. Results between each judge pair are shown below.

| Emotion        | 1--2     | 1--3   | 2--3   | Average |
| ------------- |:-----:|-----:|-----:|-----:|
|Anger             |0.43|0.50|0.44|0.46|
|Disgust           |0.55|0.51|0.49|0.52|
|Fear               |0.61|0.72|0.66|0.66|
|Happiness        |0.72|0.78|0.71|0.74
|Mixed Emotion   |0.39|0.47|0.41|0.42|
|No Emotion      |0.63|0.69|0.64|0.65
|Sadness          |0.45|0.53|0.40|0.46|
|Surprise          |0.31|0.53|0.35|0.40
|Total              |0.51|0.59|0.51|0.54|

Values differ between emotions and between judges. The biggest are for **happiness** and **fear**. This means that the reviewers on IMDb express these emotions with more clear tone using words that define them easier.
On the other had, the lowest values are for surprise and mixed emotion, and this is an expected behavior as the first is often confused with others and the latter has been added to categorize sentences that do not belong to any of the other categories.
Nonetheless even a Cohen's kappa value of 0.40 is considered a good agreement.

To finalize the dataset, I kept only sentences were at least 2 judges agreed on the category and added that label to the sentence. In total **2,002** sentences are contained in the final dataset.

## 2. Feature Selection
































