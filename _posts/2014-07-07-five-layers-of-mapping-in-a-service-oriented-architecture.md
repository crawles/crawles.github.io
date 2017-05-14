---
layout: post
title: Five layers of mapping in a service oriented architecture
published: True
feature: 

---

Pulling together a flexible service oriented architecture is a challenge.

One of the struggles we are currently having is around the various connection points between each layer.

Each layer reflects a boundary context with a set of entities, models, classes, structures. Passing these objects up or down between these boundaries add complexity. It’s a fascinating process of translating from one context to another. A given objects “meaning” is contextual…

We are now going through a process of adding another suit of tests focusing in on mapping between each context. Both “ingoing” and “out-coming”. This suit of tests is becoming more and more important when adding new functionality.

Interesting to note, the external facing contract is the most sensitive. It’s the one our clients like the most.

### Why have a workflow contract?

One of the problems we found with using .Net 4 Workflows is that it struggles to keep up with domain model changes. The Xaml files created by workflows are not as smart when it comes to renaming a class, type safety and build flexibility. If you try to rename a property don’t expect the xaml file to pick up those changes.

Given that our domain model is build straight of our edmx it’s important that it isn’t tightly coupled to the objects used within our rules engine workflows. To create this flexibility we had to create another set of objects used by workflows and map to and from these objects. We’ve called these objects Workflow contracts.