---
layout: post
title: The readify code challenge
published: True
feature: 

---

*   [Readify – Code for success](http://readify.net/)
*   [Somee.com](http://somee.com)
*   [AspSpider](www.aspspider.com)
*   [Best way to reverse a string](http://stackoverflow.com/questions/228038/best-way-to-reverse-a-string)

To get started down the Readify interview process they kick things off with an interesting challenge.

_Build and deploy a wcf service that passes their testing client application._

You can see the puzzle here: [https://knockknock.readify.net/](https://knockknock.readify.net/)

### Here’s the mind process I went through in passing this challenge.

*   What? I’ve gotta build a web service and deploy it? Hmm. I’ll use somee.com to host it.
*   Wait, am I building a web api or a wcf service? Aren’t we all RESTful now? OK fine, I’ll build a WCF Service.
*   God dang this Data Cntract. I’ve gotta use the one they provide. Hmmf. 1hr later…
*   Huh? What’s ExtensionData. Probably should provide that.
*   Performance. I remember this [maths problem] performance question when I was at uni years ago. Hmm. And what about the ReverseWords function. Perhaps that needs to be improvement.
*   And now I’m puzzling through resolving issues with somee.com hosting. Also tried aspspider.com. It’s timing out and being unfriendly. Maybe I shouldn’t have taken that /unsafe compile option. Eek.
*   And then some more. There is an interesting maths challenge I had to stumble through. I’ll leave out my thinking on that one, I’m sure you’ll discover the solution no problem.

Awesome first step interview process. Creative. Curious. I like it.

    We've run some tests against your endpoint and it's come back as scoring 53 out of a possible 100\. You don't need a perfect score to pass, but you will need higher than 53\. Have another go. :)

They mention it’s supposed to take approx 1-2 hours to complete. It took me a few hours longer than that but I enjoyed it all the same ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

    We tried to run some tests against your endpoint, but things were taking quite a long time. (We actually had to stop our tests early to prevent timeout issues!) See if you can speed things up a bit, and then have another go. :)