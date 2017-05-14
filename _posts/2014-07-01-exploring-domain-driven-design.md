---
layout: post
title: EXPLORING DOMAIN DRIVEN DESIGN
published: True
feature: 

---

*   [Domain Language](http://domainlanguage.com/)
*   [An Introduction To Domain-Driven Design](http://msdn.microsoft.com/en-us/magazine/dd419654.aspx)
*   [Domain-Driven Design: Tackling Complexity in the Heart of Software](http://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215)
*   [Anemic Domain Model](http://martinfowler.com/bliki/AnemicDomainModel.html)

D is for Domain. Not worrying about persistence for now…

## Intro

Term. 2003\. Books. DDD. 10yrs old. Practices.

Overview. Complex software project. Technical points. The value of these practices. Learning. How much more there is to learn. The unknown.

Eric Evens. Thoughtful. The Art of software.

Why use DDD? Inspiring. Repeatable. Large scale complex projects. Principles. Patterns. Difficult problems. Clean. Testable. Understand Client. Build Software. Write Code. “people don’t want to buy a 1/4inch drill. they want to buy a 1/4inch hole.”

Tackling complexity. 10,000 feet view. Domain expert collaboration. Cross over language. Sub domains. Separating out concerns. Divide problems. Modelling sub domains. Avoid implementation details or infrastructure concerns.

Benefits. Focused. Small. Autonomous. Process. Results. Flexible. Move. Side affects. Resource. Map to clients understanding. Clear pathway.

Not suitable for every project. Best for complex business needs, not complex technical problems. Eric evens quote… Sometimes overkill. Simple crud projects don’t need DDD. Domain experts need to be available.

Social collaboration. How to get others to follow DDD. Concrete benefits. Convincing others…

Navigation map.

Refactoring. DDD helps us refactor when we discover deeper insights.

## Modeling Problems

Domain export. Continuous conversation.

Language. Avoiding technical terms. Don’t use the word “class” or “table” etc. Ask the client what terms mean.

    "As software developers we fail in two ways. We build the thing wrong, or we build the wrong thing." 
    Steve Smith

Bounded Context. Helping to define a boundary within which an entity can exist within. The reuse of entities is a common practice. We need to be sure that each entity knows where it can exist. Some entities span multiple context. Other don’t. Separate team, code base and database schema. Namespaces. Folders. Projects.

DDD is sometimes vague. It’s a process of guidance. It’s something that offers options. It’s a mix of experience and guidance to make decisions. It’s a set of ideals.

Sub Domain. Bounded Context. Confusingly similar. Sub Domain is problem based area of focus. A Bounded Context is a solution based area of focus. Often these can correspond. Carpet analogy. In a fresh solution that tend to be very very similar. When dealing with a legacy system this is much more useful. A system that doesn’t match the business requirements would create very different sub domains and bounded context diagrams.

Common problem. Unclear context boundary. It’s difficult to create DDD models in a world with boundaries. This is a business / resource management / organisational problem.

Cross-cutting concerns. Venn digram. Shared Kernel. Removal of translation. DDD helps us prevent the work of translation from business to tech. Business experts. Programmers. All parties need to understand without any need for translation. Fractured language results in serious project problems. The vocabulary needs to be ubiquitous within a given bounded context. Term clarification. Industry terms.

Core Domain. Sub Domain. Problem Domain. Bounded Context. “within the context of A, …”.

#### Anemic Domain Model

Model with classes focused on state management. Good for CRUD. Very close to the database schema.

Anti pattern. Anemic classes. Simple domain objects. Anemic. Fine in a CRUD model.

Anemic. Simple domain models. At first glance they look like the real thing. Simple behaviour. Simple crud. Simple gets and sets. Limited business logic. Business manager objects sit on top of the domain model handle complex logic.

#### Rich Domain Model

Model with logic focused on behaviour, not just state. Preferred for DDD.

Rich. The goal is to create Rich Domain Entities. Domain objects with complex relationships and expanding business logic. Represent both the data and the logic of the domain.

### Domain Entities

Entities. Objects or classes that are defined by their identity. A traceable object. Entity have properties, these properties have values that change. The ID doesn’t change. Mutable properties(changing). Some Entities are created in-line with the database, others do not persist. Some entities are identified by a guid and a created whilst working through DDD practices.

Single responsibility principle… Entities should not have a lot of business logic. When designing entities we need to focus on it’s responsibility. An entities responsibility revolves around it’s identity and it’s life cycle.

Tracking identity might not be just an ID. Perhaps an ID is’t going to be good enough. Identity can be more complex. When it gets to be too complex another wrapper entity is recommended.

Comparing entities is not something that should ever happen. Checking equality is a troubling concept when it comes to entities. Is this a copy? Are these equals? What do you mean? Why would we want to check if a ‘customer’ is the same as another ‘customer’….?!

Entity relationships. One way is best. Pick verbs and see which actions are required, this should help isolate when bidirectional relationships are required. yngni – you’re not gonna need it.

### Domain Value Objects

Value. Defined by a set of values.

Measure, quantity or describe something in the domain. No simple identity. Immutable. Comparable. Unchanging. Methods won’t change the object.

string is an example value object. methods on a string return a copy of hte original string, it doesn’t change the original value.

Better to use value objects rather than entities. If you do have entities let them serve as value objects.

Value objects are a better place to hold business logic than entities. We can do our reasoning without concern for state change. Logic with no side effects.

Entity objects may have many Value objects.

Value objects are much easier to test. They are easier to use. The Entity orchestras the value objects. The entity has value objects which do all the computation. Methods of the entity may exist that pull together various value object computation methods. Entities can end up with higher level methods, use case scenarios.

If functionality can’t fit within a value object a Domain Service can be used to provide it. This is a last resort. Taking this approach too often may end up with an anemic domain model.

### Domain Services

Important operations that don’t belong to any Entity or Value Object.

Services belong in the core of the application and shouldn’t contain any business logic.

## Aggregates

A transactional graph of objects.

Aggregates and aggregate roots helps us keep complex data simple. As our data entity relationships expand and becoming more complex the runtime object graph can become unruly. To help resolve this using an aggregate pattern can help keep things isolated and simple. Bounded context.

Atomic. Grouping. ACID. Transactional.

    "An aggregate is a cluster of associated objects that we treat as a unit for the purpose of data changes."
    -Eric Evans

### Aggregate Roots

The pathway which handles the group and ensure integrity.

### Invariants

Rules.

Somethings don’t change. Some things must be true. Conditions. Must be true. To process a request a set of invariants must be evaluated.

A given entity might have a set of value objects. This entity would become an aggregate. This aggregate would ensure that a set of invariants were satisfied.

### Further Research

*   Do I have an anemic model?
*   Tell me more about the various technical terms we use when working with DDD.