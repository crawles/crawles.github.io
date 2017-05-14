---
layout: post
title: RUNNING THINKTECTURE AUTHORIZATION SERVER LOCALLY
published: True
feature: 

---

*   [JWT.IO Decoder](http://jwt.io/)
*   [Google JWT Decoder](https://developers.google.com/wallet/digital/docs/jwtdecoder)
*   [OAUTH2 Simplified by Aaron Parecki](http://aaronparecki.com/articles/2012/07/29/1/oauth2-simplified)

Wow. After much faffing I’ve now had great success with stepping through the Resource Owner Authentication and Authorisation flow.

Where did I struggle?  
1\. Watch the videos. Success.  
2\. Get the source code for Identity Server, Authentication Server and the Sample Flow Clients.  
3\. Sorting out a Self Certified SSL Certificate.  
4\. Setting up Trust for the certificate.  
5\. Sorting out signing keys and other configuration issues.  
6\. Running a flow locally.

**Resource Owner Flow**  
For logging in with a username and password. The Resource Owner Password Flow is really pretty simple, as it allows the client to exchange a user’s username and password for an access token.

**Implicit Flow**  
For browser-based or mobile apps. You should use this flow for client-side web applications (JavaScript clients) that need temporary access (a few hours) to the user’s data. In Implicit flow,the token is returned directly via redirect URL.

**Code Flow**  
For apps running on a web server. This flow includes sending the client user via redirect to the provider’s login and authorization page, then will redirect back to your web application and pass a authorization code in the URL parameters.

**Client Flow**  
Client credentials for application access. Some API endpoints does not require an account access token since they have to be executed without an authenticated user

Still got a long way to go with wrapping my head around all the ins and outs of OAUTH2\. Fascinating stuff!