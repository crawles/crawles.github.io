---
layout: post
title: "Semver is a powerful example of agreement and wide spread collaboration."
categories: []
tags: []
published: True
feature: "semver.jpg"
caption: "jontejada.com"
---

- [ASP.NET Community Hangout - January 19th, 2016](https://www.youtube.com/watch?v=FSf83_TU5Yg)
- [Semantic Versioning](http://semver.org/)
- [GitVersion](https://gitversion.readthedocs.org/en/latest/)
- [Jake Ginnivan](http://jake.ginnivan.net)
- [How we version our software](http://andreasohlund.net/2013/10/29/how-we-version-our-software-in-particular/)
- [Why semver ranges are literally the worst…](https://medium.com/@kentcdodds/why-semver-ranges-are-literally-the-worst-817cdcb09277#.r0b7kxxg8)
- [Semantic Versioning Basics on Youtube](https://www.youtube.com/watch?v=QMUSkra7Blk)

Semver is a pretty quite topic. For what's there to help us with I'm surprised I'm only now learning about it. It's the sort of thing that can help on many different levels.

## What is Semver?

Semantic Versioning is a commonly used naming convension for labeling a library, component, package, api, service or any other shared piece of technology. It can also be used for document store management or anything else that is constantly evolving.

The specification is authored by Tom Preston-Werner, inventor of Gravatars and cofounder of GitHub. It details the meaning of a given version number. From looking at the commit history I can see that semver was beginning to formalise in early 2011.

	"^3.10.1"

Semver specifies what "^3.10.1" really means. At it’s simpliest it implies [major].[minor].[patch] release details.

This standard is becoming more and more important in the progressive open source web based development world we all live in. And it also brings into play our new found ability to handle transitive dependency.

Versioning is a wide and complex problem space and semver creates a language on which we can formalise simple agreements.

Are you using it in your current project?

> "Lets say you update versions of libraries your project depends on and make no other changes. When you re-release your project, which version number should you bump? I think it ought to be the minor number and that it shouldn't depend on whether those libraries broke backwards compatibility. I'm curious what others think?"

It's these sorts of questions semver is setup to solve. 

Without a common understanding of how we label our releases it becomes increasingly difficult to allow for a healthy ecosystem to take hold. 

Semver is one awesome "agreement" success story that allows a massively diverse group of people to work well together.

Thinking in semver has also started to change the way I think about my backlog!

If you're working on a bunch of components that want to play happy together then take a look at how semver can help you play better together.