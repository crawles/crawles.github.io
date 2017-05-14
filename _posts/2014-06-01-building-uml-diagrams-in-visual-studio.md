---
layout: post
title: BUILDING UML DIAGRAMS IN VISUAL STUDIO
published: True
feature: 

---

*   [Solution Modeling with UML in Visual Studio](http://pluralsight.com/training/courses/TableOfContents?courseName=vs2010-uml&highlight=richard-seroter_vs2010-uml-module1*10,11#vs2010-uml-module1)
*   [UML in Visual Studio](http://channel9.msdn.com/Niners/clinted) by Clint Edmonson

Been working at collaborating with business analysts, solution architects and other developers and constantly drawing diagrams.

_I did not realise that visual studio ships with a suit of tools that can make this sort of work a whole lot easier._

Here’s how I went about figure out how to draw up a sequence diagram using Visual Studio.

### UML

Diagrams in Visual studio are drawn up using UML.

    "The Unified Modeling Language (UML) is a general-purpose modeling language in the field of software engineering, which is designed to provide a standard way to visualize the design of a system." -wikipedia

## Use Cases

    Use case diagrams show the relationship between users and their goals when using a system.

Brain storming. Capturing initial requirements. Define boundaries. Higher level system concepts.

<table>

<thead>

<tr>

<th>Component</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Actor</td>

<td>Represent users. Human users or non human users. “Cashier”. “Mobile phone”. The initiator. One actor might be a more specific instance of another actor. a “Casher” **is an** “Employee”. Actors don’t interact with one another.</td>

</tr>

<tr>

<td>Use Case</td>

<td>A circle with text. Business language friendly. Single behaviour. Allows for ‘inclusion’ of other common use cases.</td>

</tr>

<tr>

<td>System Boundaries</td>

<td>Sub systems. Illustrates all the use cases that fit within a given sub system.</td>

</tr>

</tbody>

</table>

## Activities

    Activity Diagrams show the business processes or system algorithms.

How describe the ‘flow’ rather than the ‘how’. Made up of decisions and Actions.

These sorts of diagrams can be drawn up using UML Activity Diagrams or Windows Workflow Activities.

## Sequences

    "Easier than others. Non technical. Sequence diagrams show how components interact over time."

<table>

<thead>

<tr>

<th>Component</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Lifeline</td>

<td>Model the objects that participate. The vertical line. Top down.</td>

</tr>

<tr>

<td>Message</td>

<td>The line going between each lifeline. Synchronous or Asynchronous.</td>

</tr>

</tbody>

</table>

The most exciting aspect of sequence diagrams is the simple way of nesting reusable sequences with reference sequences!

Hmm. Wow. I think sketching interfaces with sequence diagrams is going to consume a lot of my time going forward.