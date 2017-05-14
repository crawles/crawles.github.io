---
layout: post
title: WEB API 2 AUTHORISATION HERE, HERE & HERE
published: True
feature: 

---

Keywords: Clients. Users. Filters. Attributes. Roles. Permissions. Claims. Scopes.

Authorisation helps us deal with the following issues.  
What happens after we get access to the Api. How do we take a given identity and access their authorisation. How do we allow them to access certain resources?

**User Authorisation**  
The most common type of authorisation. Identity. Claims. DB Profile.

**Client Authorisation**  
Trust clients? Confidential client? Securely authenticate the client?

You might want to limit the user based on the type of client they are using to access the resource. Hmmm?

**Where does authorisation actually happen?**  
1\. OWIN middleware.  
2\. Global Authorization Filter.  
3\. Controller Authorization Filter.  
4\. Action Authorization Filter.  
5\. … in your code.

At each of these the steps the process can be stalled and given a 401 Unauthorized.

Avoid mixing Authorisation logic with Facade or Business logic.