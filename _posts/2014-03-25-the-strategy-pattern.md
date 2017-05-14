---
layout: post
title: THE STRATEGY PATTERN
published: True
feature: 

---

When should it be used? When you want to track the ‘state’ of an object as it passes through assessment or maturity.

e.g. Is your Package.Entity “Proposed”, “Active”, “Resolved”, “Closed”… ? Are you tracking an object through a series of known steps? If we are having to regularly check the state of an object to decide on the course of action, the state pattern can help!

If we are tracking the state with a hard-coded condition we have the following design concerns.  
– All states must be known and agreed up.  
– Not easy to add new states.  
– If we object has it’s own set of conditions they are all attempting to deal with these ‘progress’ concerns separately.  
– Difficult to unit test. Each method, in each object, with a large set of conditions has a complex set of outcomes.

**How does the State Pattern help us?**  
– It aims to change the behaviour of the object by altering it’s state.  
– Encapsulate the logic of each state into a single object.  
– Improve testability by running tests for a given state without having to worry about how other states may be affected.  
– Dynamic state discovery. The organic discovery of new states and new behaviour for these new states – even at run time!

The immediate benefits are clear. This separates concerns for each state. Simplifies state transition. Improves testability.  

<div class="simplePullQuote">

Use the state pattern whenever the methods of an object need to change depending on it’s state.

</div>

Here are some other Behavioural Design Patterns worth exploring.  
– Chain of Responsibility Pattern  
– Command Pattern.  
– Mediator Pattern.

Configurable!  
– Strategy Pattern, Change behaviour based on configuration.  
– State Pattern. Change behaviour based on state of object.

Rules Management  
– Specification Pattern. Create a set of rules that helps filter or isolate a particular outcome. The process of acting on the outcome is not taken into account.