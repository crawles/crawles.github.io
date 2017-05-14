---
layout: post
title: ACCOUNTS WITH MICROSOFT.ASPNET.IDENTITY.CORE
published: True
feature: 

---

Some quick notes on the new MVC5 security model that came out a few months ago.

When working with the default ‘Individual Accounts’ authentication template one of the nuget packages installed is the microsoft.aspnet.identity.core package.

This assembly defines some core interfaces that are used by mvc and webapi to manage the persistence of identity.  

<div class="simplePullQuote">

For MongoDB, XML, or any other type of persistence these interfaces need to be implemented with an alternative approach

</div>

**IUser**  
Defines a user with Id and UserName.

**IRole**  
Defines a role with an Id and a Name.

To persist user information there are a variety of interfaces to be implemented. These types of objects fit within a Repository type design pattern.

**IUserStore**  
Provides a set of functions that allows the Creation, Deletion and Modification of a given User by Id.

**IUserLoginStore**  
Used to validate a given users login credentials.

**IUserPasswordStore**  
Used to manage a given users Password.

There are some concrete classes.

**UserManager**  
Takes an object that implements the IUserStore interface. Used to validate usernames and password, hashes passwords. Validates identities.

**Role Manager**  
…

For SQL server these interfaces have already been implemented.