---
layout: post
title: THE CHAIN OF RESPONSIBILITY
published: True
feature: 

---

Consider a set of processor objects are organise in a line. The first object is handed a message. They assess the object and either process it or pass it on. If the message is not processed by the current object it is passed along.  

<div class="simplePullQuote">

I have a Expense Claims process that passes the expense claim along the staff hierarchy chain until it is assessed and approved. I keep adding new types of staff members and reorganising the staff hierarchy?! And sometimes the printer randomly does the assessment – Wtf!

</div>

So why not just have a loop and ask every member of staff? Good question. That decision to ask every employee is a big one. It’s a decision that is not too flexible. Would it be more flexible if each person who was asked could implement their own solution to the question – Who should we ask next? Extrapolating that decision out to each person involved is one way the chain of responsibility can help.

The word Responsibility is key with this design. It’s very important to consider that with this design pattern the order in which the components are ordered is notable. If your object structure has a natural hierarchy, this approach will be really useful. Without a hierarchy, would a simple list do the job?

Use the Chain of Responsibility to help manage a diverse set of objects built to ‘assess’ an incoming package in relative order.

Anyone faffed around with an Asynchronous Tree of Responsibility?