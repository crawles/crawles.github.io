---
layout: post
title: GOOD ARCHITECTS AVOID MAKING DECISIONS
published: True
feature: 

---

*   [Clean Architecture and Design](https://www.youtube.com/watch?v=asLUTiJJqdE) by Uncle Bob

Astrology. Laser beams. Science and all that no computational joy. And now then, clean architecture ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

### Addicted to picking technology

Picking a technology stack is not an architecture. Architects should avoid picking tools. Tools like Entity Framework, .Net, XML, Web Api, JSON, SQL, MVC, … these are all tools. The architecture is not the tools.

Picking the tools is the easiest step of the project. It looks great but it doesn’t achieve anything. To get something to work the real challenge is figure out what needs to be built. Figuring that out is technology agnostic.

There are frameworks, design patterns and standards. It’s important to follow these standards. But, following these common industry approaches isn’t the challenge. Picking these standards again isn’t the challenge. Picking tools, frameworks and design patterns is not what an architecture should be doing.

### Self explanatory architecture

The architecture should of a building helps us understand what something is. It tells you how to use something.

The architecture of a church or a library tells you what it is and how to use it. This is what architecture does.

    "This building looks like a church. That's cool architecture!"
    - Uncle bob

Often an architecture document doesn’t tell us what an applications “is”. Instead, our architecture documents often tell us a list of tools.

### What are the use cases?

If an architecture doesn’t focus on use cases it’s simply can’t tell you what it does.

An architecture is a picture of use cases.

When you look at the architecture of a system and all you can see is a web server, a model diagram and various layers then the architecture is hiding the use cases.

Again, what are the use cases?

### Forget the delivery mechanism

The mechanism for delivering a set of functionality is not an architecture. Again this is a set of tool.

_The internet is just another example of a delivery mechanism. This is not of any concern for an architect._

The delivery mechanism is a diagram of techology. This sort of approach doesn’t make it easy to organise use cases. A technology stack doesn’t help us discuss use cases.

The delivery mechanism is a bolt on. This is non architectural detail. The ‘web’ is a detail.

### Avoid making decisions

A good architecture allows us to get started without having to make major technical decisions. The data model, the database tools, languages, orm’s, C#, the web, … these sorts of decisions are not required when coming up with an architecture.

A good architect maximises the number of big decision NOT made for as long as possible.

Some big upfront decisions can be deferred right off the end of the project.

### So where to start?

1.  Use cases.
2.  Business rules.

Don’t let the technology stack rule your design. With a more informed understanding of the business we are better at picking the right technology stack.

### Real architects write code.

Good architects write code and they understand the pain involved in working on the system.