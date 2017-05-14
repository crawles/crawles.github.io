---
layout: post
title: WHAT REST IS AND ISN'T
published: True
feature: 

---

*   [Richardson Maturity Model](http://martinfowler.com/articles/richardsonMaturityModel.html)
*   [REST Fundamentals](http://www.pluralsight.com/training/Courses/TableOfContents/rest-fundamentals)

We can’t get enough of this stuff at work.

Perhaps the goal of this post is to prove to business folk that talking about rest is not a business level concern. It’s a system level technique with awesome potential.

Every 10 minutes I hear this… “Is that sort of approach RESTful?”

What does it mean to build a web product that adheres to RESTful principles?

## What REST is.

*   representational state transfer.
*   rest is an architecture style. some artists use an impressionist style when they create a picture. Some architects use a RESTful style when they create a modern distributed system.

## What REST isn’t.

*   Rest is not a business requirement. Rest is a system level technique used to create a service that can be used by a business to achieve its goals.
*   it’s not a component or library approach.
*   it’s not a new way to use http to make remote procedure calls via soap. With rpc the target is to provide a set of actions for a given entity. With rest the goal is to provide the entity itself on which the client can carry out required actions. This involves more complex semantics for the client application which is a key goal of restful design.
*   rest is not http. rest can be built on any protocol.
*   rest is not Uris. Hyperfocus on Uris can make a service not restful. Focusing on Uris is an easy way to slip back into an rpc approach.
*   rest is not a network protocol.
*   rest is not an object level design pattern.
*   rest is not soap. Soap is an implementation detail. A rest service could potentially use soap as part of its implementation.

## Why bother with REST?

Here are 3 reasons why REST has become something we want to talk about when building a modern distributed system.

### Heterogeneous Interoperability

*   Network based rather than library based api.
*   Independant evolution and progression. Evolvability and long living services goes to the core of rest. Independant component release management.

### The rise of devices

*   different operating systems.
*   different views on the same resource with different features and demands.

### The Cloud

*   service level scalability.
*   cloud computing is only useful because when done correctly it can save you lots of money.
*   restful services utilise the benefits of a cloud architecture / scalable infrastructure.
*   services hosted in the cloud need to be elastic.

## Rest Maturity

A service becomes RESTful by passing through the following 4 phases of maturity. Once you hit level 3, your system can now be classified as RESTful.

### Level 0 : The swamp of plain old XML

The starting point for the model is using HTTP as a transport system for remote interactions, but without using any of the mechanisms of the web. Essentially what you are doing here is using HTTP as a tunneling mechanism for your own remote interaction mechanism, usually based on Remote Procedure Calls.

### Level 1 : Resources

The first step towards the Glory of Rest in the RMM is to introduce resources. So now rather than making all our requests to a singular service endpoint, we now start talking to individual resources.

### Level 2 : HTTPVerbs

Using the HTTP verbs as closely as possible to how they are used in HTTP itself.

### Level 3 : Hypermedia

It addresses the question of how to get from a list open slots to knowing what to do to with any given entity any number of actions or processes. Self documenting responses.