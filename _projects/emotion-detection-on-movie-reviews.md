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

##Emotion Detection on Movie Reviews

The object of this project is the **Emotion Analysis** of sentences that are comming from movie reviews, using **Machine Learning**. An attempt will be made to construct a
**classifier** capable of classyfying a sentence in one of the 6 basic categories of emotion which are **anger, disgust, fear, happiness, sadness, surprise**, and the category of **non-emotion**.

Things I did include:

- Created a new **dataset** using **inter-annotator** agreement
- Applied **feature selection** with **tf-idf** and **chi-square** methods
- Performed 15 **text pre-processing techniques** and their combinations
- Created and tested 14 new features in **feature engineering**
- Applied **ensemble learning**
- Used the resulting classifier in 6 movies

##1. Creating the Dataset

The [IMDbPY](http://imdbpy.sourceforge.net/) was used, a free Python package for the retrieval and manipulation of data from the largest movie database [IMDb](http://www.imdb.com/).
Movies with more than 40 reviews were randomly retrieved and their reviews were stored in a file. Subsequently, representative keywords were chosen for each emotion category. These words can be seen in the following table.

| Emotion        | Keywords           |
| ------------- |:-------------:|
| Anger     | angry, annoyed, gloomy, depressed, unhappy, down, disheartened, sorrowful, painful, guilty |
| Disgust     | stupid, sucks, irritated, humiliated, disgusted, nauseatng, sickening, contempt, repelling, unpleasant      |
| Fear | afraid, frightened, fearful, horrified, nervous, paniched, alarmed, phobia, scared, insecure      |
| Happiness | awesome, happy, amused, fantastic, excited, pleased, cheerful, love, great, amazing      |
| Sadness | sad, lonely, gloomy, depressed, unhappy, down, disheartened, sorrowful, painful, guilty      |
| Surprise | astonished, bewildered, survrised, confused, sudden, unaware, shocked, perplexed, what, unexpected     |











