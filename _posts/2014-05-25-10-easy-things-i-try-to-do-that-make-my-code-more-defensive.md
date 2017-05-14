---
layout: post
title: 10 things I do that make my code more defensive
published: True
feature: 

---

Here are 10 quick and easy things I try to do every day that make my code more defensive.

## Single Purpose Methods

When I write a new method, or when I find myself modifying a method, I keep in mind that the method should do just **one** thing.

If I’m refactoring the code and adding an additional logical step I’ll try to split the method up into two separate methods.

## Separate Concerns

Here’s an example of how to ensure Event type concerns are kept separate from Business concerns.

When I create a click event handler or a controller action in a web application I avoid putting any entity focused logic into the handler.

        public IHttpActionResult SignUp(string userName,string email,string password)
        {
            var account = Customer.Signup();
            // avoid putting any 'signup' relevant logic in the handler of the event.
            return Created();
        }

Here’s a quick list of commonly mentioned concerns I try to keep separate.

<table>

<thead>

<tr>

<th>Concern</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Presentation</td>

<td>The layout. What people see.</td>

</tr>

<tr>

<td>User Interface</td>

<td>The interactive elements of the Presentation layer.</td>

</tr>

<tr>

<td>Action Handling</td>

<td>What happens when a user interacts with an interface.</td>

</tr>

<tr>

<td>Service Integration</td>

<td>How the client application integrates with the service.</td>

</tr>

<tr>

<td>Message handling</td>

<td>How to take a request and manage it’s progress through the system.</td>

</tr>

<tr>

<td>Business Entity Management</td>

<td>How the key entities within the system are managed.</td>

</tr>

<tr>

<td>Business Logic</td>

<td>Containing reuasble busines logic functionality.</td>

</tr>

<tr>

<td>Business Rules</td>

<td>Reusable business conditions.</td>

</tr>

<tr>

<td>Business Actions</td>

<td>Reusable business actions.</td>

</tr>

<tr>

<td>Data Context</td>

<td>How the application access the database.</td>

</tr>

<tr>

<td>Data</td>

<td>How the data is stored.</td>

</tr>

</tbody>

</table>

## I write code that I will want to read next year

I write code to my future self. I have a 5hit memory. I forget stuff.

When I write code I aim to organise it in a way that I’ll be able to comprehend in the future.

## I try to write DRY code.

    DRY. Don't Repeat Yourself.

When I find a line of code that is repeated twice on the same screen I start to question my approach. It usually reflects a broader issue that can be resolved fairly easily with a little refactoring.

## I write unit test.

I start with a failing unit test, than runs quick!

The top 3 rules of writing defensive code are.

1.  Write unit tests.
2.  Write unit tests.
3.  Write unit tests.

I write tests that defend my code through time. I write tests that are intended to pass, fail and throw exceptions. I write unit tests to speed up my red, green, refactor cycle. I write unit tests to improve the design of the application.

## I avoid magic numbers.

I don’t add constant values to my code.

    var pageSize = 10;
    ... or ...
    var organisationId = 123;

If my method needs a constant value that means I need to think a little harder about my implementation. Does it really need a constant value? Will it ever change? How to I keep it in a safe place that can updated later?

## Short & Sweet Logic

If a piece of logic is rambling on I’ll reassess my approach.

    "If a method scrolls beyond the height of my screen then it's too long!"

## I avoid the blame game by rejecting the garbage.

Software is like an orchestra. Lots of musicians performing together. A balance.

The challenge is to find a way to add to the symphony without getting entangled in the mistakes of others.

To avoid getting stuck I try to add ‘guard statements’, throw useful exceptions and end the flow of the method as soon as possible.

Rather than creating flexible methods that can take ‘everything’ it ends up being easier in the long run if I write methods that are specialised and fail cleanly if the incoming request is slightly smelly.