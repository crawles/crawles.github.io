---
layout: post
title: TDD with Jasmine and AngularJS
published: True
feature: 

---

*   [Sample Plnkr](http://embed.plnkr.co/uXWs8jYtrj8BDHpURRfp/preview)
*   [Testing AngularJS From Scratch](http://pluralsight.com/training/Courses/TableOfContents/testing-angularjs-from-scratch)
*   [Jasmine – Behavior-Driven JavaScript](http://jasmine.github.io/)
*   [AngularJS](https://angularjs.org/)

It’s easy to adhering to the 3 laws of TDD in a well contained C# / Java middle tier development world. Getting that sort of workflow balance in the Javascript / Html world is a new challenge.

Jasmine and AngularJS make it possible to build complex JavaScript **test driven** logic and closely tie that up with a clean Bootstrap user interface.

Getting into a spacious Red, Green and Refactor loop on the front end is out goal.

### More notes on TDD

Have you ever noticed that it takes longer to find the red phase than any other?

Why is it so important to find that clear red phase? Why is it so important to actually write a concise test that fails?

It’s so important so you can confirm that it was broken before you fixed it. It proves that “what you’ve just done to fix the problem” does actually fix the problem. Amoung many other great benefits, it helps us avoid the ambiguity of unexpected behaviour and context variation.

It’s a real challenge but it’s important. Only write enough code to make the test pass. No more. No less. Just enough.

Unambiguous. Fast. Immediately obvious what just broke. Implied guidance on resolution.

Don’t write tests that check code you didn’t write.

I just like that line. The idea is that we jump from one failing test to the next seeking out the most interesting use case for each step ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

### BDD and AngularJS

ok great. we’ve now got a bunch of js classes (oops, i mean functions) behaving well – with a clean suite of tests. Now, how do we get those tests moving inline with our ui angular work…?

angular and jasmine integrate fairly simply. It’s a quick way to write a suite of tests that hit on models, controllers and services.

Here’s an example of two unit tests that check if a controller fires up as expected.

    describe("Farm Controller", function () {

        var $controllerConstructor;
        var scope;
        var controller;

        beforeEach(module('JasmineGarden'));
        beforeEach(inject(function ($controller, $rootScope) {
            $controllerConstructor = $controller;
            scope = $rootScope.$new();
            controller = $controllerConstructor('FarmController', { $scope: scope });
        }));

        it("check if farm controller can be instantiated.", function () {
            expect(controller).toBe(controller);
        });

        it("check if farm controller has animals.", function () {
            expect(controller.Animals.length).toBe(3);
        });

    });

### Type safety

Again. I’m still overly fussed about the issue of javascript type safety. I’m writing unit tests to check simple things like values exist but it really doesn’t help me figure out what is really going on under the covers.

_Is anything undefined in those models that were instantiated once the data came back from the service?_