---
layout: post
title: Sydney's continuous delivery crew
published: True
feature: 

---

*   [Continuous Delivery Sydney July Meetup](http://www.meetup.com/Continuous-Delivery-Sydney/events/181191512/)

Last night the Sydney Continuous Delivery crew hosted a meetup. Here are my notes…

## The Vodafone Continuous Deployment Reformation

### Sydney is a great place to work in tech

As the event began it was clean the turn out was impressive. The seats were full, the booze flowing and the finger food steady. Cba did an very nice job of providing the space.

### Why can’t we deploy at midday?

The problem is this. Releasing. software is a complex job. It’s the sort of thing that used to happen at 3am. A series of seemingly unrelated manual tasks with unknown points of failure. It was perhaps the worst part about the job. Vodafone had been feeling the pain. So they invest in a continuous deployment strategy.

The first half of this event was learning about vodafone’s CD transformation.

### Infrastructure as code

The first step they made was managing the provisioning of environments with code. Using AWS. Spinning up their various environments with scripts in the cloud. Dynamically generated cloud formations. Minimal configuration differences.

This fast ability to fire up environments makes it possible to spit apart teams, project and timelines without too much hassle. It’s surprising how stubborn architecture influences business decisions. Avoiding this is our goal.

Creating environments and spaces to experiment on “close to production” type environments allows for a more flexible work place.

### Automated Testing

This is both the hardest part of the CD story and also the most important. Without this it’s simply not possible to confidently achieve CD.

Automated unit testing, integration testing, acceptance testing, pre prod testing and post deployment testing are all fundamental to the spirit of CD.

### Branching and Feature Switching

The problem is this. How do you work on the code base for an extended period of time? How to check out the code and keep it checked out until it’s ready. Some things take a number of days to complete.

Vodafone found two approaches work well. Branching. Feature Switching.

Feature switching involved working of the main branch of the code and simply not enable to component of work until it was deemed ready. By then, it was already in production. Switching on/off functionality at runtime. DNS switching.

### Zero Downtime

Yep. That’s right. Zero down time. Finding a way to release software at midday with zero down time is the goal. It’s possible.

### When done is done

When the code hits production. If it’s not in production it’s debt. If it’s not on the shelves in the supermarket, it’s inventory in the stock room. It’s waste.

One of the most significant benefits to CD is Developer empowerment. Giving developers a space to work and confidently release to production has compounding benefits.

### Deploying should be boring

Deployments are a non event. Anti climatic. Boring. They should be regular. Daily. Hourly.

### Fail safety

Expect things to fail. Region failure. Be ready. Resilient. Visible. Health checks. Disks run outs. Automated recovery. Kill the world so the problem is irrelevant. Auto healing. Auto scaling.

### And more…

Splunk. New Relic. Developer empowerment. Stateless. Security. The unknown database story.

## Microsystems with Thoughtworks

Building systems that pivot.

Micro services is basically another way of saying – SOA, RESTful, project autonomy, collaborative development. The great benefits of working within a RESTful development paradigm are well known. The concept of Microservices really isn’t much of a technical breakthrough. Instead the word “Microservice” is a development guidance principle that fits within an agile software development life cycle.

### Pets. Farm Animals

Treat your servers bad. It’s interesting to consider that with the cloud we have new ways of designing systems. We also have new ways of thinking about our servers.

Rather than treating your servers like pets, treat them like farm animals. Treat your servers like cattle. Have lots. Kill them frequently.

### Managers want something in 6 months

It’s interesting to hear that, from an industry survey…

    "At least 60% of managers expect to see their ideas in production within 6 months."

Continuous deployment and micro services makes this easier to achieve.

### The architecture elephant in the room

We talk about building good software. We talk about unit testing. We talk about performance. We talk about design patterns. There’s something else we developers need to be talking about…

Architecture. Architecture. Architecture. Architecture. Talking about an architecture. Architecture. Elephants.

Single box. Layered box. Fail, layer teams. Difficult to works in a single tier. Away from client. Affects release cycle. Conflicting interests. Syncing teams.

Break by business function rather than technical function. Each service should be able to chose its own tech stack. Language agnostic api. A simple minimal integration layer. Open standards.

Composability. Lego. Replaceable. Hard to decommission. More important than reuse. Splitting. Security. Settings. People, team structure. Corza sore…? Each service needs to own it’s own data. Avoid a service bus, a fat band across the middle. Perhaps use the rest maturity 3? SOA. Micro services is another Verison of SOA. Wsdl. Ws*. SOA never had an agile manifesto.

### Clojure

Clojure? To be discovered…

### All up? All down?

We have this problem often. People want to know. Is the system working? Is the system “up?”.

Building systems along those lines is destined to fail. Instead we should be focusing in on systems that are decoupled from an architecture perspective. All up? All down? Mostly up? Mostly down?

    Don't rely on anything. There is no SLA for an ec2.

## Sydney Tech Meetup

Sydney tech meetups are a great way to make friends, learn and find mentors. Sharing knowledge in the tech industry is a difficult skill. Going to these industry focused events makes it easier.