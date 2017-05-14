---
layout: post
title: Here's why recursion is so important when we start to think about functional programming.
published: True
feature: 

---

*   [F# Sydney](http://www.meetup.com/fsharpsydney/)
*   [Functional Programming – Lots of emphasis on recursion, why?](http://stackoverflow.com/questions/12659581/functional-programming-lots-of-emphasis-on-recursion-why)
*   [The Value of Values with Rich Hickey](https://www.youtube.com/watch?v=-6BsiVyC1kM)
*   [Robert C. Martin – Functional Programming: What? Why? When?](http://vimeo.com/97514630)
*   [Structure and Interpretation of Computer Programs](https://mitpress.mit.edu/sicp/full-text/book/book.html)

Immutability. When we write software that relies on static components it unlocks a world of multi processor opportunity. Rather than working hard to push objects through varying states we can start to focus on expression and logic instead.

For those of us who have been writing object oriented imperative software for the past decade a resurgence of the word “Recursion” might seem a little strange. Its a dirty word in an imperative world. But learning about how functional programming relates to recursion leads us down an interesting path.

At the core of what makes Functional Programming so great is it’s re-focus on avoiding side affects. Recursion makes it possible to deal with collections without having to write an assignment statement. This is a significant shift in thinking. With a few simple tricks functional programming deals with the usual performance and stack overflow concerns.

Recursions is an old school technique that seems to shed new light in a multi core processor future. It helps us avoid the Von Neumann memory allocation bottle neck. It helps us make our software easier to write, more flexible, capable of greater complexity, expressive, more efficient, easier to test, and concise.