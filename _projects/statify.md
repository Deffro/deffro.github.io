---
title: "Statify"
date: 2020-11-14
author_profile: true
excerpt: "An app where anyone can join with Spotify credentials and see a lot of cool statistics about his/her profile"
tags: [pandas, visualization, flask, dash]
header:
  image: "images/projects/statify.png"
  teaser: "images/teasers/statify.png"
categories:
  - Data Visualization
  - Software Development
  - Web Development
  - API
  - Deployment
mathjax: "true"
---

An app where anyone can join with spotify credentials and see a lot of cool statistics about his/her profile.

Before visiting the app, you might want to check the Known Bugs section at the end of the page.

Since the app uses a lot of requests, does a lot of processing and creates many visualizations, it can take up to 20 seconds to load.

**The project is available on my [GitHub](https://github.com/Deffro/statify).**

## <a href="https://statify-deffro.herokuapp.com/" target="_blank">Click Here to Visit the App</a>
(if the message "error loading dependencies" appears, it is a known bug and just refresh the page)

<img src="/images/projects/demo.gif" alt="Statify-Demo" />

## Technologies/Concepts used
- Spotify API
- Object Oriented Programming
- Git, GitHub
- PEP8 notation
- Pandas
- Plotly and Dash
- Flask
- Ngrok
- Html, Css
- Heroku

## How to run it by yourself
- Create an app in [spotify](https://developer.spotify.com/dashboard/applications) and generate your client id and client secret
- Copy and paste them in server.py
- Run ngrok.exe and type ngrok http 8080
- Copy and paste the ngrok url in CLIENT_SIDE_URL in server.py
- Visit again your spotify app, go to settings and under Redirect URIs paste the ngrok url. Also add "http://127.0.0.1:8080/" there. Don't forget to save
- From terminal run: python server.py
- In a browser visit the ngrok url

## Known Bugs
- If the message "error loading dependencies" appears, it is a [known bug](https://github.com/plotly/dash/issues/125) and just refresh the page.
- If you visit the site and after you another person in the world also visits the site, then if you refresh the "/app/" page you will see his statistics and not yours. If you want to see your statistics refresh the index page. The current state of the app is not to be refreshed. 
















