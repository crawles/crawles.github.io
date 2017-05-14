---
layout: post
title: Releasing Musomap mobile 2.0.4
published: True
feature: 

---

*   [Musomap](http://www.musomap.com)

I’m a .NET guy. Sometimes side project drag me other ways.

For example; Musomap. A crazy personal hobby “global map of musicians” project. Its built with MySQL, PHP, HTML, JQuery, JavaScript, Json. When I started the project nearly two years ago I didn’t know what I know now. E.g, AngularJS.

The Musomap mobile app uses the same technology, and releases via Phonegap. It works fine for what I want it to do.

With Musomap’s social reach it does continue to connect local musicians and promote local events. I can’t just turn it off. People keep joining and finding a reason to use it. It services their needs, it costs me money and time to run. I enjoy it.

Last week I decided to show it some love and do a small clean up. Copy changes, layout fixes, that sort of thing. The functionality of the appseems to be stable and useful.

Developer ownership. It’s a constant journey to find the minimal viable product.

Running this sort of volunteer, not for profit, spare time project really has taught me a lot about progress and servicing the needs of the client. Development. Release management. Communication. Consistency. Feedback.

People sign up first on the website. I thought that mobile apps were now lead the way…? With Musomap it seems people first learn / experiment with the info on the website. Only once they trust the service do they go mobile. I mean, did you sign up to Facebook or Twitter on the website or the mobile app?

Anyway, the updated Musomap mobile app is now out. Version 2.0.4\. I hope you like it.

The release process involves some fairly nimble moves through a cloud9 online editor, php scripts deploy to UAT, store changed files, zip web app, build on Phonegap, download IPA and APK files, upload to android console and itunes connect (via Macincloud). Google take a few hours to release, Apple take 3 days to release.

I’m yet to automate the release process.