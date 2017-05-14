---
layout: post
title: Loose coupling and high encapsulation with Udi Dahan at Microsoft Sydney.
published: True
feature: udi.jpg

---

*   [YOW! Nights: Supporting the developer community. The only independent tech conference in Australia.](http://nights.yowconference.com.au/)
*   [Particular Software](http://particular.net/)
*   [@UdiDahan](http://www.twitter.com/udidahan)

This week Readify hosted a top notch tech evening in collaboration with YOW! Nights, Particular and Microsoft.

Our very special guest speaker was Udi Dahan.

Udi is one of the world’s foremost experts on Service-Oriented Architecture and Domain-Driven Design. He created NServiceBus. My good mate Hadi is part of the team. They do awesome stuff.

Udi speaks in a dark, dry, confident, straight forward and humourous way. He’s a captivating software developer.

The evening left me with some notes.

## Questions

In the Q&A I sheepishly stuck my hand up and asked him two questions. He handled them well.

> “You mentioned Dimension. A stack is one dimensional. Up and down. The idea that one dimension is an unsuitable way to think about complex problem. I find that fascinating. Keeping contracts and entities simple and divided. Doesn’t that become an issue when even in a given service the direction of the message affects the attributes. As in… the customer object “in” is different to the customer object “out”. What sort of techniques have you found can help with this contradiction of definitions?

Udi mentioned an interesting approach around not returning “everything” and instead splitting up services more finely. A sequence of calls that play together.

I then went on to ask…

> “Often our clients ask us to _please try to reduce the numbers of calls to the server_. We end up meeting that need by compromising ont he simplicity of any given end point. We basically mash a whole bunch of stuff into one space. How do you prevent that “get everything into that first request” scenario?”

Udi mentioned two different thoughts. Infrastructure needs and network limitations are a key player in your design. They always need to be taken into account at a high level. He also mentioned that sometimes it’s reasonable to mash a whole bunch of stuff into a single request – but that doesn’t mean you have to compromise on design. There are techniques for doing this clean.

## “Loose Coupling & High Encapsulation”

At one point Udi nearly had us chanting “Loose Coupling & High Encapsulation”.

Again. Nothing new. But he did shed light on what most dev buzz words actually mean.

> “SOA = Service Oriented Architecture. The O and the A actually means nothing. the S – makes some sense. It’s all jardon. All these words are a business way of saying something very simple.”

Say it with me… “Loose Coupling. High Encapsulation.”

## Naming Things. Nouns. Verbs. Attributes.

Udi proposed that one of the hardest things about writing clean code was the process of figuring out what something is “called”. This issue goes to the core of DDD. I agree. Terminology, understanding, agreement, interpretation and assumptions are the demise of most things – including software development.

His advice was simple.

> “Don’t start with trying to figure out with the Nouns are. That will take months. Go with the Verbs and Attributes first. The Nouns are no important. They will come later. They will come with maturity.”

## Layers. Friction Points.

Udi raised his concerns around layers. He mentioned the “stack” mentality of “up and down” created an arena in which complex problems are just much harder to deal with.

> “”i hate layered architecture. the lines. the lines are connections. connections are mad. those lines hurt. lines are coupling points.”

I interpretted that to mean – a good soccer team doesn’t have lines – just a group of highly encapsulated loosely coupled agents. Make sense.

I tend to agree. Layers are a troubling over simplification.

## Thanks!

The beer. The friends. An excellent night. See you all again soon!