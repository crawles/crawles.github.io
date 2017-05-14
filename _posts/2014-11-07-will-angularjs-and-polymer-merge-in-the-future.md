---
layout: post
title: Will AngularJS and Polymer merge in the future?
published: True
feature: 

---

*   [All About Polymer at SFHTML5](https://www.youtube.com/watch?v=mN7IAaRdi_k)
*   [What is the difference between Polymer elements and AngularJS directives?](http://stackoverflow.com/questions/18089075/what-is-the-difference-between-polymer-elements-and-angularjs-directives)

A question was asked at Google IO this year…

> “Will AngularJS and Polymer merge in the future?”

Interesting. With my recent adventures with handlebars, angular, underscore, polymer, backbone, ember, web components, AMD and various other front end technologies this question touches on a subtle point. This question highlights a conceptual gap.

How should we be thinking about mvc and web components?

AngularJS has done an excellent job of building an incredibly effective framework. Despite recent issues with their latest release it still remains the most popular JS framework in my space.

With web components the way we perceive a problem is very different. Web components involve a completely different design and strategy from the DOM up.

There are many different ways to ‘separate concerns’. MVC is one approach. Web components is another. They share few similarities. The best way to isolate the differences is to step through a basic demo.

Angular has high level APIs for services, filters, animations, supports IE8, and at this point, is a much more robust framework for building production apps. Polymer is just starting out in alpha.

To give you an overview. Working getting started with Polymer & Web Components these are the sorts of things you’ll need to learn;

*   Using HTML imports.
*   Using Polymer elements with standard HTML, CSS and JavaScript.
*   Creating a custom element using Polymer.
*   Working with shadow DOM.
*   Templating and Using data binding.
*   Publishing properties.
*   Declarative event handling.
*   Adding properties and methods to the element’s prototype.
*   Automatic node finding.