---
layout: post
title: IS TDD DEAD?
published: True
feature: 

---

*   [Is TDD Dead? Part 1](https://www.youtube.com/watch?v=z9quxZsLcfo).
*   [Is TDD dead? Part 2](https://www.youtube.com/watch?v=JoTB2mcjU7w).
*   [TDD is dead. Long live testing](http://david.heinemeierhansson.com/2014/tdd-is-dead-long-live-testing.html).
*   [Why Most Unit Testing is Waste](http://www.rbcs-us.com/documents/Why-Most-Unit-Testing-is-Waste.pdf).
*   [Thought Works](http://www.thoughtworks.com/).
*   [David Heinemeier Hansson](http://twitter.com/dhh).
*   [Kent Beck](https://twitter.com/KentBeck).
*   [Martin Fowler](https://twitter.com/martinfowler).

“TDD as One True Way” versus “TDD as devil-spawned tempter” is not a productive contrast.

On 09 MAY 2014 Thought Works hosted a Google Hangout. Within a few days over 16,000 people have watched the recording on YouYube. There are two follow up discussion on YouTube that continue the conversation.

The 3 speakers where David Heinemeier Hansson, Kent Beck and Martin Fowler. No boxing. No conflict. Just 3 guys grappling with an important issue. TDD.

Here’s what I took from the debate.

### What is TDD?

Amidst the conversation the concept of a TDD became clearer. One of the goals of the discussion was to reduce the scope of what TDD actually means.

Here are some statements about TDD that helped me refine my understanding of what it is supposed to mean.

*   TDD is often described as a Red, Green Refactor Workflow.
*   TDD is one approach that can help a code base achieves it’s optimal health.
*   TDD is not always useful or helpful.
*   Using TDD is a question of trade-offs.
*   Unit testing and TDD are closely related. If something can be easily unit tested then TDD is going to be your best friend.
*   TDD applies a certain pressure on design freedom. Don’t let TDD influence design.
*   TDD works well when you have a lot of experience with the requirements and the correct design. When the implementation is smelly, it’s difficult use TDD. _Perhaps the proper design will never not be achieved on this project…?_

### What is a Unit Test?

The definition for “unit test” is a slippery sucker.

    "A unit test provides a strict, written contract that a piece of code must satisfy." -wikipedia

I’d like to prose this.

    "A unit test is a strict written contract that a piece of code, with no dependencies, must satisfy."

And as you can imagine, I’m struggling with the obvious problem with this statement. What piece of code could ever have ‘no dependencies’???

My answer? using design patterns to pull apart dependencies. (more on this later but, it comes as a surprise and a relief to find out that using a mocking framework is also NOT recommended). Another dead end.

The real world example I’ve been grappling with is unit testing an MVC Web Api 2 Controller. I’ve made my notes on the issue [here](http://www.kahneraja.com/unit-and-integration-testing/).

I’m now starting to realise that for most basic full stack MVC applications it’s more useful to abandon unit testing in the initial phase. Integration testing in a world where nothing is a much faster way to get automated testing. TDD offers a faster feedback cycle and a higher code base health. There are many others approaches that can help us achieve this too!

### How TDD can lead to bad design decision.

What? TDD leading to bad design decisions? But isn’t TDD here to help us conceptualise the needs and lead to loosely coupled, easy to read, easy to maintain well designed code?

Hmmm. Well. It’s interesting to hear David clearly articulate his issues with how TDD can often lead to implementing overly complex and difficult to understand implementations.

*   TDD occasionally encourage rambling, non-obvious designs to take shape.
*   TDD leads to an Alice in Wonderland of mocking.
*   With requirements constantly changing, TDD creates baggage that inhibits flexibility.

I’m starting to realise that an initial focus on automated Integration Testing would resolve many of these issues with TDD.

When you’re dealing with design decision in the initial phase, don’t let TDD influence your approach. TDD is most useful after the key design moves have been made.

At the end of the day, some snippets of code may not be covered by a unit test and that’s OK! The challenge is to ensure that these snippets of code remain as simple as possible.

    "Don't let TDD influence design decision." -Kent Beck

### Is Mocking dead?

Wow. It’s quite amazing to hear so much anger towards the various mocking frameworks. I absolutely agree with the notion. I’ve founding mocking to be a scary rabbit hole of endless magic inventions.

It’s very interesting to consider that extensive mocking really does bring into focus the validity of the tests being written. Are we testing the system or are we testing this unrealistic mock?

I’ve found a nice rule that was recommended.

    "When mocking an object, only setup a single layer. I tend not to go beyond that. The trade-off is simply too high."

#### Positive flow and confidence.

A large amount of the discussion was about flow and confidence.

    "Is a developer allowed to feel confident?"

Kent mentioned that he loves to satisfaction of working towards a clearly defined goal. The Red, Greed, Refactor loop is something is adores.

The conclusion I took from the debate on flow was significant. I found that if the component your writing can be easily encapsulated and examined without dependencies then writing unit tests and using test driven developer is going to make your life a lot easier.

If on the other hand you’re working with in an MVC model where the model, view and controller and heavily linked then perhaps separating out the business logic into a separate object is a good idea. With this approach you might consider writing Integration test for the MVC components and Unit Test for the newly created Business Object.

This is going to be easy to unit test.

    public class Order
    {
        public decimal CalculateOrderTotal(Order order)
        {
            decimal total = 0;
            foreach(var item in order.Items)
            {
                total += item.Price;
            }
            return total;
        }
    }

This is NOT going to be easy to unit test because it depends on an entity framework and a database. Instead an automated integration test would be more suitable.

    public class Customer
    {

        public decimal SaveOrder(Order order)
        {
            Context.Customer.Orders.Add(order);
            Context.SaveChanges();
        }
    }

Find a way to configure the application so it runs different when running a suit of integration tests. comprehensive automated integration testing is going to keep test driven development alive. They are two separate approaches that work well together.

#### The TDD cult.

The discussion at work is heated. Every young professional strides in with the agenda of pushing TDD. Manages love these cats. Manages love the visibility of unit test.

It’s now comforting to see how seasoned developers are finding a clear voice on how TDD is good in some cases and not good in others.

#### Code base health.

I left the debate with a renewed focus on code base health.

It was great to have David, Kent and Martin agree on one thing. If TDD can help you built code base healthy then great!

Our goal here is to build a long lasting healthy code base.

What is the health of your code base right now?