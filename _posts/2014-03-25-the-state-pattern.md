---
layout: post
title: THE STATE PATTERN
published: True
feature: 

---

Arguably one of the simplest design patterns out there. A great place to get started.

**How is this pattern going to help me?**

*   <span style="line-height: 1.714285714; font-size: 1rem;">It helps to encapsulate a related set of algorithms. A family of calculations or assessments.</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">Improve Separation of Concerns Principles and Open/Closed Principles.</span>

**Strategy Pattern Smells?**  
Smells! A smell in code is a bit of code you think smells real bad.  

<div class="simplePullQuote">

I have various ways of calculating the price of a product, they vary depending on the selected shipping method. I keep adding new ones and it’s getting crazy! Hmm…

</div>

When it comes to the Strategy pattern the following smells might be solvable…

*   <span style="line-height: 1.714285714; font-size: 1rem;">Lists of conditions or growing switch statements.</span>
*   <span style="line-height: 1.714285714; font-size: 1rem;">Adding another similar ‘calculate cost’ method with a small variation. Another family member – an ugly uncle who drags the family down with all their boring nostalgia about transport and stream trains. meh.</span>

The strategy allows us to encapsulate each algorithm into a separate concrete implementation that can easily be expanded upon.

Show Interfaces some love, and they will love yo back.