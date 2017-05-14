---
layout: post
title: MY TOP 10 FAVOURITE RESTful PRINCIPLES
published: True
feature: 

---

*   [What REST is and isn’t](http://www.kahneraja.com/what-rest-is-and-isnt/)
*   [REST Fundamentals](http://www.pluralsight.com/training/Courses/TableOfContents/rest-fundamentals) by Howard Dierking.
*   [Hypermedia to the Rescue](http://msdn.microsoft.com/en-us/magazine/jj883957.aspx) by Pablo Cibraro.

I’m a C# developer and here’s my TOP 10 RESTful Principles.

## Why RESTful is still awesome

Given the changing internet landscape, RESTful gives us a set of principles to work with when building a modern distributed system.

Services now need to reorient their architecture to account for the rise of devices and cloud computing.

### The rise off devices

Once upon a time it was acceptable to build a system for a set of constrained devices and a single operating system. Things have changed.

The client application is now king.

The rise of devices brings two major concerns into focus.

*   How to provide different ways of viewing and using the same data.
*   Cross platform collaboration (aka heterogeneous interoperability).

REST principles are closely tied to servicing the rise of devices.

### Cloud computing

Cloud computing is a new way of hosting application. It’s surprising how cheap a server can cost.

Modern distributed systems needs to be built differently if they want to take advantage of cloud computing.

Smaller cheaper nodes doing simpler work makes this possible. The significance of scalability, reliability and service evolution are all close tied to REST.

## My Top 10

<table>

<thead>

<tr>

<th>–</th>

<th>Principle</th>

<th>Tag</th>

<th>Summary</th>

</tr>

</thead>

<tbody>

<tr>

<td>1.</td>

<td>Hypermedia</td>

<td>No need for a predefined contract.</td>

<td>Hypermedia decouples the client from the server. Hypermedia makes it possible to rebuild a set of Uri’s without having to deal with changes on the client. Hypermedia allows the server to drive the discovery of resources and activities. A good example of hypermedia is HTML. The txt in HTML comes with a load of mean. HTML is full of additional attributes and placeholders for additional action meaning. For example, a Form comes with an Action attribute. Hypermedia allows us to add a new step in a workflow without an automated client being aware. “The client made an additional uri request because it was told to do that _next_…”. It’s excellent when a RESTful client only needs to know a single resource uri to get started.</td>

</tr>

<tr>

<td>2.</td>

<td>Best friends with HTTP</td>

<td>Web maturity.</td>

<td>Http comes with great stuff. Content data, conditional get definitions, rich status codes, cache headers, Accept headers, verbs, authority tokens, query string parameters, non txt format data, location header. Treating the rules of http as a first class solution player makes things so much easier! Rest doesn’t try to reinvent the relationship management skills http has gained through maturity.</td>

</tr>

<tr>

<td>3.</td>

<td>Stateless</td>

<td>Let’s keep things simple and scalable.</td>

<td>A single restful action really does live in its own little stateless world. It’s chatty. This means that each action can be dealt with independently. This makes it easier for it to be improved, updated and versioned. Without having to manage state, each server node is allowed to flake without catastrophic results. Not having to manage ‘who the client is’ on the server makes solution architecture so much more flexible.</td>

</tr>

<tr>

<td>4.</td>

<td>Uniformity</td>

<td>Let’s join with other RESTful players.</td>

<td>Rest works well when it allows a system to become a player in a community of other services. It allows a service to be easily picked up by various devices that also depend on other RESTful services. Abiding by a uniform interface means we can collaborate.</td>

</tr>

<tr>

<td>5.</td>

<td>Cross Platform</td>

<td>You don’t need to be Microsoft to join the party.</td>

<td>Um. Yeh. I think this one speaks for itself. Who wants to build a different way of dealing with different clients.</td>

</tr>

<tr>

<td>6.</td>

<td>Single responsibility</td>

<td>Keeping each resource action simple.</td>

<td>Decoupling the server from the client and keeping each action simple makes it possible to more easily modify server/client workflows and expected parameters.</td>

</tr>

<tr>

<td>7.</td>

<td>Network Caching</td>

<td>Why pay for a caching solution when the Internet already does that?</td>

<td>For requests not running across SSL, the ability to use the network to handle various caching opportunities means RESTful services can run on cheaper resources. Rest leans on the network benefits, it gets that stuff for free. It means some level of caching can be achieved outside the solution.</td>

</tr>

<tr>

<td>8.</td>

<td>Visibility and Responsiveness</td>

<td>When a client steps through a Workflow we can keep them updated with their progress.</td>

<td>The Internet is made up of packets. The end of the bulk upload black box. Real-time feedback. It’s chatty and that’s an excellent thing. It keeps the implementation simple.</td>

</tr>

<tr>

<td>9.</td>

<td>Maturity</td>

<td>There are 3 levels of maturity before you can graduate with a RESTful certificate.</td>

<td>To figure if a solution can be theoretically classified as RESTful there’s a guide. Level 0: The swamp of plain old XML. Level 1: Defining “Resources”. Level 2: HTTPVerbs. Level 3: Hypermedia.</td>

</tr>

<tr>

<td>10.</td>

<td>External Representation</td>

<td>Business logic & domain model flexibility.</td>

<td>The ability to decouple domain model from external representation creates flexibility. The domain model can change. The domain model must change. What doesn’t change are the external facing Resource models. These are business focused and don’t change! There is a noisy separation. RESTful Resources exist on the internet and take on a life of their own.</td>

</tr>

</tbody>

</table>

## Challenges

These top 10 RESTful principles add loads of great stuff but they do come at a cost.

*   A comprehensive hypermedia implementation is rarely achieved.
*   Thinking in OO or RPC terms inhibits RESTful design. It’s a different problem space. Classes and methods won’t help us here.
*   Building RESTful clients involves more work than a regular RPC client.

## Terms

<table>

<thead>

<tr>

<th>Term</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>RPC</td>

<td>Remote Procedure Call. A way of thinking about a call to a distributed system that adheres to a design that is very common and does not adhere to RESTful principles. RPC is most likely more commonly used that REST. Perhaps RPC is more mature and perhaps out dated.</td>

</tr>

<tr>

<td>Workflow</td>

<td>In this case we are refering to a client/server workflow that involves various requests in a sequence. E.g. imagine a process of signing up which involves more than one step…</td>

</tr>

<tr>

<td>HTTP Verbs</td>

<td>OPTIONS, GET, HEAD, POST, PUT, DELETE, TRACE, CONNECT</td>

</tr>

<tr>

<td>REST</td>

<td>Representational state transfer. A software architectural style.</td>

</tr>

</tbody>

</table>