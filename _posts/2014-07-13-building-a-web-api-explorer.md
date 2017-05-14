---
layout: post
title: Building a web api explorer
published: True
feature: 

---

At work we run various web api applications. RESTful services. Lots of uri’s. Lots of contracts with client application.

To help share the knowledge of what these services I’ve created a web api explorer client using reflection. This solves that problem of constantly having to update your testing client application as it evolves.

It’s a friendly alternative to using Fiddle, Simple JS Client, and more informative than the standard .Net Help page.

This project makes it easy to publish to the outside world everything your RESTful service can do!

> [github.com/kahneraja/ApiExplr](http://github.com/kahneraja/ApiExplr)

This web api explorer is made up of two components. – 1\. A library for exporting your web api controller into a json feed. – 2\. Consuming that json feed and publishing it into a friendly website that allows you to make requests to every controller and action.

### Rendering the Json

Getting it to trawl your web api layer and return it in friendly Json just isn’t easy.

I had to build a fresh controller that returns the required json from the ground up using regular reflection. Search out all your controllers, actions, parameters and model graph yourself.

### Building the client

Using AngularJS and Bootstrap we can quickly parses that json feed and create a client that can give your controllers and actions exactly what they want.

*   Sort out a basic runthrough on plnkr.co
*   Setup AngularJS routing.
*   Build a data service to get our Json feed.
*   Rendering a ‘tree’ with nested input boxes matching the incoming Json.
*   Dealing with collections.
*   Integrating Token Authentication.

### Always happy to help

Please don’t hesitate to get in touch by posting a comment below. Let me know how this could be improved!