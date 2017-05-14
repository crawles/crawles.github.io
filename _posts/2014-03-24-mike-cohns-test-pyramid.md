---
layout: post
title: MIKE COHN'S TEST PYRAMID
published: True
feature: 

---

**“But I really hate unit tests. Every time I do anything I have to spend most of my time rewrite endless test?! – Not worth it!”**

Mike Cohn devised the notion of a testing pyramid. It’s a way to isolate different types of test and assign effort to each field. It’s a great to way to avoid spending too much time checking functionality manually.

Before I get started on all that, here’s why automated testing is great!

*   <span style="line-height: 1.714285714; font-size: 1rem;">Shorter Feedback Loops. Automation shortens the time between implementation and feedback.</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">Lowering the Cost of Change. Automated tests give us a safety net to re-factor the system.</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">Self documenting</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">Proof of requirements.</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">An alternative TDD approach that helps keep development simple and easy to share.</span>

So, back to the pyramid…  
The test pyramid describes how automated tests should be distributed among different types.

By the time a product owner gets their hands on the UI to run a few test most of the works should have already been done – and proved successful. User interface tests are very hard to develop and maintain, they run slow, and they tend to be very brittle. For every UI test their should be a corresponding larger set of test for each layer below.

At the service level it is easier to develop automated reusable tests. it tends to be more robust but there are still many dependency concerns. Test often commit data encapsulated dependencies. Services tend not to embrace Open/Closed principles.

Unit tests that assess the success or failure or a single unit of work are much easier to maintain. These tests tend to suit a Business Objects. With easy implementation and a low total cost of ownership a well organised collection of unit tests will help drive the sustainability of any expanding system that needs to adapt with business needs.

Interestingly, with the rise of thick clients (mobile, JavaScript libraries) and simple services, perhaps some of these ideas might need to be flipped on their head. Either way, the notion of a single units of work being automatically & regularly tested remains important.

**How to build stuff without having to refractor a world of legacy test?**  
Refactoring code and making changes is an inevitable part of adapting to business requirements. Sometimes this will need to happen. But there is something interesting to consider here.

When a test needs to be rebuild we have an opportunity. An opportunity to sit back and look at why these test needs to be rewritten. Is this test directly associated with the changes required? Is it far off and unrelated? These questions lead to a conversation about the way our code has been written? If an unrelated test need to be rewritten this is loud and clear comment on the tightly coupled, unsustainable nature of the architecture and development. This is something that needs to change and it’s great that the test has highlighted the underling design issues.

An agile development approach with founded design patterns is the answer to – “why am I rewriting all these tests.”