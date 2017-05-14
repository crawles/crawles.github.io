---
layout: post
title: 12 things you can do to get developers writing more (better) unit tests
published: True
feature: 

---

*   [Martin Fowler: Test Pyramid](http://martinfowler.com/bliki/TestPyramid.html)
*   [Is TDD Dead?](http://martinfowler.com/articles/is-tdd-dead/)
*   [10 reasons you should NEVER write a unit test](http://www.javacodegeeks.com/2013/10/10-reasons-why-you-should-not-write-unit-test-cases.html)
*   [Uncle Bob: TDD Part 1](https://cleancoders.com/episode/clean-code-episode-6-p1/show)
*   [Pluralsight: The world’s largest tech & creative training library](http://pluralsight.com/)

Is the unit test debate over baked? Maybe…

I’m a TDD fan boy who regularly shares Martin Fowler’s Test Pyramid in heated debates over maintainable tests. More often that not my unit testing ideals are met with frustration.

> “TDD looks great on paper but doesn’t work in-the-real-world…”

That aside, if your team has _decided_ to go with TDD then here are 12 things you can do to get colleagues and loved ones writing more unit tests.

### Training

More training.

TDD is difficult if you haven’t been given the time to really explore the toolset.

Robert Cecil Martin has made an art form out of talking about fundamental software development principles and practices. He has two excellent video sessions on why TDD is worth its weight in gold. His presentation is curious, funny and challenging. If you have a colleague who likes on-line training videos, try this!

Pluralsight have a number of excellent hands on video courses for TDD at all levels of your application stack. Backend. Frontend. Server side. In the browser. And beyond.

Enjoying the benefits of unit testing requires training.

### Lead by example

Get cracking on TDD in your own development. Wait and see how others slowly but surely start to rely on your unit tests.

It’s difficult to convince others of the merits of unit testing. It’s difficult to prove that TDD will save you time, encourage better design, or ensure quality. Avoid the debate.

Instead, come up with tests that others will find useful. Fix something with unit tests that no one else has been able to fix. Prove the case by making life easier for your colleagues.

### Developer empowerment with continuous delivery

Continuous delivery and automated release management is only possible if everyone involved build and deploys with a suite of tests that build trust.

The dream of lunchtime deploys is only possible if everyone plays their part in writing test at all levels of the application.

Writing tests means you don’t have to deal with manual deployments at 3am.

### It’s not the gate keeper

Developers write unit tests. QA people ensure the system works. Developers are Creators. Testers are Protectors. They are friendly foes. They are trying to build trust.

Unit testing is something that will improve the lives of your developers but it will not convince Testers that the system is happy and healthy. If they are to remain employed and valuable, testers will need to duplicate this work and run their own automated quality assurance. It’s a sort of partnership. Double entry book keeping.

### Don’t force it

Unit testing is not something that offers immediate benefit to others in the surrounding team. Yes it may well ensure quality, speed, flexibility, collaboration and design but, well, maybe not.

TDD is something that simply makes your work space cleaner, easier to maintain and less stressful. Keep in mind, it’s a personal choice.

Unit testing is not something you can enforce. It’s a self respect, team collaboration and trust building exercise.

### Documentation

Unit tests are, in my opinion, the best way to explain a system.

This relates to training. Knowing how to organise, name, arrange and optimise unit tests is an important part of keeping unit tests easy to read and understandable.

When writing unit tests we are able to focus on a single feature and challenge it. This is concise.

    WhenCranky.ShouldBeQuietWithFood()

### Allow for extra time

If a colleague asks for extra time to write unit tests, give it to them!

The pay off for writing unit tests is worth it. It most cases it will actually lead to a massive saving of time. But, initially, create that space in the schedule.

### Wait until something needs to be refactored

Unit tests build confidence when refactoring code.

In the short term unit tests may not offer much value. Over time, as code needs to be updated and changed, unit tests will allow you to make changes without worry about the fall out. If you work on a fairly stable code base unit tests will be invaluable when you need to refactor individual components.

### Reward your TDD fans

Introduce a team building reward that benefits developers who write ‘good quality tests’.

Tests = Beer.

### Flow

The biggest cost to software development progress is ‘distraction’. Unit tests help retain flow.

A developer who struggles with being distracted might benefit from using unit tests to keep their working context in mind.

### Architecture

It’s difficult to write unit tests on an application that is tightly coupled. Sort out the architecture first.

Getting developers to write unit tests on a system that doesn’t already have an easy to navigate suite of tests is difficult. Get that sorted first.

### Organise your tests

Get the right tests in the right place. Unit tests. Integration tests. Acceptance tests. Developers. Testers. Who’s writing what? Where? Who’s initially responsible for these various forms of tests?

This often requires some intense discussion and debate. Having a clear shared understanding of where a certain test should go is important. The Test Pyramid can help.