---
layout: post
title: 3RD PARTY AUTHENTICATION WITH MVC5
published: True
feature: 

---

*   [Securing RESTful Web Services with OAuth2](http://blog.cloudfoundry.org/2012/10/09/securing-restful-web-services-with-oauth2/)

Along with the standard MVC Entity Framework User and Roles Management, you can also integrated 3rd party external authentication. When people login to an MVC5 web application it’s very easy to integrate with protocols like OpenID and OAUTH2\. This means people can login to with their Facebook or Twitter credentials easily.

Some popular external authentication providers are: Microsoft Live, Facebook, Twitter, Google, …

There are two advantages to using 3rd Party Authentication.

**1\. Developers**  
Developers don’t have to manage password and password resets. There is less of the messy details around credential management. You can let the 3rd party handle all of that for you!

**2\. Users**  
User relying on their 3rd party account benefit from using a single account to access various websites. This is more secure.  

<div class="simplePullQuote">

One thing to consider is, do you trust that 3rd party authentication provider?

</div>

There are some extra hurdles to consider. For example, which each 3rd party provider you’ll need to setup and maintain an active developer account and rely on the key and secret they provide. If their service or api changes you’ll need to adapt too.

The complete working user flow for Google, Twitter, Facebook and Microsoft Live Account are already built and seamlessly integrate with MVC5\. WebApi 2 is included in this. WebApi2 relies on MVC5.

**Steps Involved In Kicking off a 3rd Party Example**  
1\. Check out Startup.Auth.cs. Uncomment the segment that you’d like to enable. The simplest line to start with is app.UseGoogleAuthentication();. This integration doesn’t require any developer key setup. Uncomment, and go!  
2\. Fire up your app.  
3\. Login (with Google). Implicit OAuth Flow. You can see the screenshots of this approach below.

When this 3rd party authentication has been approved our local database generates a set of values that allows our application to recall and map to the 3rd party authentication details. The two standard tables that were used to persist this 3rd party credential details were. AspNetUserLogins and the standard AspNetUsers

This is a great way for small public websites to be built quickly.

**Owin Middleware**  
Owin is a industry wide web application / hosting development standard. It’s code. Katana is Microsofts implementation of that standard.

When we implement this 3rd party Google Authentication we are ‘enabling’ another Owin component. app.UseGoogleAuthentication() is adding another layer to our Katana pipeline of requests. This allows our authentication to be handled before the request reaches our controllers.

Generally. Owin middleware sits independantly in a pipeline. Each component can process a given request and/or pass it on to the next component in the chain. Each component in general is unaware of one another. Owin components are light weight and tend to work in isolation. This pipeline is defined in Startup.cs.

The cookie is managed by a middleware component. It takes the cookie and populates an identity for the given request. Has it expired? What roles do we have?

The issue is this. Sometimes we need to access the ‘context’ of our middleware components whilst running within our subsequent controllers. E.g. when we get to our callback url from our 3rd party client we need to capture our login details to step through the process of sign out, sign in, logging in etc.

**OAuth Claims**  
When working with 3rd Party Providers Claims help us define a person. “I claim that this person is KahneRaja. I claim that this users email address is kahneraja@gmail.com.” etc. To get access to a users email address on callback, we need to review the additional claims provided.

These claims are organised into a different way for each provider. Each claim requires permission approval that is apparent when the users authenticates.

<a href="">![Fire up application from Visual Studio.](http://www.kahneraja.com/wp-content/uploads/2014/04/1-150x150.png)</a> <a href="">![Login.](http://www.kahneraja.com/wp-content/uploads/2014/04/2-150x150.png)</a> <a href="">![Select Google account.](http://www.kahneraja.com/wp-content/uploads/2014/04/3-150x150.png)</a> <a href="">![Set Permissions.](http://www.kahneraja.com/wp-content/uploads/2014/04/4-150x150.png)</a> <a href="">![Callback and confirm registration.](http://www.kahneraja.com/wp-content/uploads/2014/04/5-150x150.png)</a> <a href="">![Authenticated Homepage.](http://www.kahneraja.com/wp-content/uploads/2014/04/6-150x150.png)</a>