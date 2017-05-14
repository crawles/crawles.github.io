---
layout: post
title: N-TIER THINKTECTURE SERVICE ORIENTED ARCHITECTURE
published: True
feature: 

---

*   [Thinktecture.AuthorizationServer](https://github.com/thinktecture/Thinktecture.AuthorizationServer)

We’re implementing an enhanced version of Thinktecture AuthorizationServer at work and it’s excellent. 

<div class="simplePullQuote">

We’re working on improving it’s ease of use and testability.

</div>

We have a WCF Service Layer and a Business Layer in between the TokenController and our DB Context. It’s a multi tier solution that requires a clearer separation of concerns.

Validation of Requests, Creation of new Access Tokens and utilising Refresh Tokens for each type of Request and Flow are currently bound together.

I’m looking into introducing a new interface that will help encapsulate these different types of Requests. A new IOAuthRequest interface.  
IOAuthRequest.Validate()  
IOAuthRequest.CreateToken(Request)  
IOAuthRequest.RefreshToken(Request)

This should help simplify things. We’ve put together the basic Access Token and Refresh Token implementation. i’m now finding that i’m spending quite a bit of time pulling apart the implementation in order to make further enhancements and write unit tests. Things are too tightly coupled and wild.

For example…  
We are not using standard OAUTH2 Scopes. Instead, each controller action has a unique identifier. Users are granted permissions to individual actions through an additional db lookup in an Web Api 2 Authorisation Filter.

Rather than creating a TokenRequestValidator and assess the Grant_Type we could instead process each request with a new ITokenRequest implementation.

To be continued…