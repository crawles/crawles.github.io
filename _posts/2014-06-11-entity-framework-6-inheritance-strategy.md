---
layout: post
title: ENTITY FRAMEWORK 6 INHERITANCE STRATEGY
published: True
feature: 

---

*   [How to choose an Inheritance Strategy](http://blogs.msdn.com/b/alexj/archive/2009/04/15/tip-12-choosing-an-inheritance-strategy.aspx)
*   [Inheritance with EF Code First: Part 1 – Table per Hierarchy (TPH)](http://weblogs.asp.net/manavi/inheritance-mapping-strategies-with-entity-framework-code-first-ctp5-part-1-table-per-hierarchy-tph)

There are 3 different approaches to implementing inheritance with Entity Framework.

<table>

<thead>

<tr>

<th>Name</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Table Per Hierarchy (TPH)</td>

<td>All data for a given hierarchy in one table.</td>

</tr>

<tr>

<td>Table Per Type (TPT)</td>

<td>Each type has it’s own table. The base class has the shared attributes.</td>

</tr>

<tr>

<td>Table Per Concrete Class (TPC)</td>

<td>Each type has it’s own table. The database doesn’t share fields across entities.</td>

</tr>

</tbody>

</table>

Each approach has it’s advantages.

### Table Per Hierarchy (TPH)

Table Per Hierarchy is generally better performing, because no joins are necessary, everything is in one table.

### Table Per Type (TPT)

Most commonly used. Allows for a more flexible design.

### Table Per Concrete Class (TPC)

Has some issues with working within the VS EDMX IDE.