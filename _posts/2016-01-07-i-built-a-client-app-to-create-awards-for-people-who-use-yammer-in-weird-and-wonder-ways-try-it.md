---
layout: post
title: "Yammer Awards. I built a client app to help us create awards for people who use Yammer in weird and wonderful ways."
categories: []
tags: []
published: True
feature: "yammer.jpg"
---

- [Yammer Awards Client App. Try it!](http://kahneraja.com/YammerAwards/dist){:target="_blank"}
- [Yammer: Enterprise Social Network](http://www.yammer.com){:target="_blank"}
- [Github Repository](https://github.com/kahneraja/YammerAwards){:target="_blank"}

Do you use Microsoft's enterprise social network Yammer?

To get us started I've now puzzled together the following (slightly weird) prizes:

- Bang For Buck Award: Most engaging post in under 140 characters.
- Group Creator Award: Creator of most popular new group.
- I Like You Award: I hit like more than anyone else.
- Silent Assassin Award: I regularly login. I never post.
- Contributor Award: I get involved by posting and liking more than anyone else.
- Batman Award: I only ever post at night.

[Try it on your yammer network and see who wins :)](http://kahneraja.com/YammerAwards/dist){:target="_blank"}

# How?

Pulling together 1000's of messages, users and groups into a single web browser isn't super easy.

I had to figure out the following:

- How do you **compress** all that yammer data into a browser window so it doesn't fall over?
- How do you **query large multi dimensional data sets** without a backend?
- How do you handle **authentication** and then host it so others can use it?

# Answers

React. Underscore. NPM. Gulp. Github pages.

	// node packages.json
	{
	  "name": "Yammer Awards",
	  "version": "1.0.0",
	  "description": "",
	  "main": "index.js",
	  "author": "",
	  "license": "ISC",
	  "dependencies": {
	    "babelify": "^7.2.0",
	    "bootstrap": "^3.3.6",
	    "browser-sync": "^2.11.0",
	    "browserify": "^12.0.1",
	    "ejs": "^2.3.4",
	    "express": "^4.13.3",
	    "gulp": "^3.9.0",
	    "gulp-ejs": "^1.2.2",
	    "gulp-express": "^0.3.5",
	    "gulp-live-server": "0.0.29",
	    "jquery": "^2.1.4",
	    "jsoncomp": "^1.6.1",
	    "jsonpack": "^1.1.4",
	    "lz-string": "^1.4.4",
	    "react": "^0.14.5",
	    "react-dom": "^0.14.5",
	    "reactify": "^1.1.1",
	    "underscore": "^1.8.3",
	    "vinyl-source-stream": "^1.1.0"
	  }
	}

Tell me. What other prizes do we need?
