---
layout: post
title: Would it be possible to build a javascript library that remembers all my CSS pain points?
published: True
feature: 

---

*   [Is there a tool to find html elements affected by certain css file](http://stackoverflow.com/questions/15472967/is-there-a-tool-to-find-html-elements-affected-by-certain-css-file)

Is this a dumb idea?

On of the project I’m currently working on is an ever expanding html prototype application. We have 100’s of simple flat aspx pages full of html. We’ve got a bunch of css files that are 1000’s of lines long.

We work through adding new pages and juggle the reuse of css. It’s a fairly simple UI centric project.

Sometimes we add or edit CSS. When we do this we are uncertain. We are uncertain of subtle changes that might be showing up on other pages. Often there are quick run around to see if any pages have been adversely affected.

In general we have strategies that help us avoid this by structuring our css intelligently but occasionally it’s unavoidable. And I want supporting with being ‘sure’!

#### Would it be possible to build a javascript library that crawls a site, making notes of every element, it’s layout features and records this in a progressive history. This might make it easier to assess what pages and dom elements have been modified over the past few hours or days.

For example. When I add the following style to a global css file I’d like to know which pages and elements have been affected.

    .confirm-btn {
        color: red;
    }

One easy way to check this would be to create an iframe, visit every page in a directory and then query every element with a class attribute using jquery.

    var $e = $('.confirm-btn');
    var color = $e.css('color');
    addThisToMyMagicCssHistorySoICanHighlightChangesLater($e.id, color); 
    // i could do something similar for all important layout features and elements for every page. Color, Height, Weight, Padding, Margin, Border, Background, ... whatever is needed to be tracked.

> “In the last 2hrs the following pages have been affected by your css changes….”

_Note: i’m not just looking for a tool that tells me where a css class name has been mentioned in a given set of pages or templates. i want something smarter than than. I want a tool that can check if stuff has actually changed. in the example above it might be the case that the .confirm-btn was already ‘red’, in such a case I’m not concerned. I want to be certain associated element has actually changed. It’s a subtle difference but it would help a lot with blanket testing and certainty._

Does such a tool already exist? A CSS Change Affected tool? Keywords: CSS. Crawl. Remember. Javascript.

Hmm. How would this actually work with a SPA?! Would it need to click through every single link across the entire website, wait for each page to load, run it’s process and then carry on? would that be too slow? Could this be done in parrallel using a headless browser?

Crawling dom elements. Is JQuery the best tool for that? Do I need an MVC framework?

#### I’m stuck. Here are my show stopper problems.

*   there’s no efficient way to crawl 100+ page websites. I’m unable to click through every link and wait around for each page to load. too slow feedback loop.
*   there’s no easy way to get every piece of html to shop up. once the page loads the ui shifts depending on how you use it. what you enter data into a form or click on something templates are rendered etc. there’s no easy to way to get every piece of html showing up in every possible way.