---
layout: post
title: SIGNING AND ENCRYPTING A JWT OAUTH TOKEN
published: True
feature: 

---

*   [Principals for OAuth2.0 implementation](http://10.102.37.20:8090/display/REF/Principals+for+OAuth2.0+implementation)
*   [JwtSecurityToken Class](http://msdn.microsoft.com/en-us/library/system.identitymodel.tokens.jwtsecuritytoken(v=vs.114).aspx)
*   [Understanding JWT with Atlassian](https://developer.atlassian.com/static/connect/docs/concepts/understanding-jwt.html)

We are using Javascript Web Tokens as a standard for formatting our Access Tokens. JSON Web Token (JWT) is a compact URL-safe means of representing claims to be transferred between two parties. JSON Web Token (JWT) is a compact URL-safe means of representing claims to be transferred between two parties. The claims in a JWT are encoded as a JavaScript Object Notation (JSON) object that is used as the payload of a JSON Web Signature (JWS) structure or as the plaintext of a JSON Web Encryption (JWE) structure, enabling the claims to be digitally signed or MACed and/or encrypted.

### Examples Token

    {
      "iss": "SSO_Server",
      "aud": "MyFavouriteApplication",
      "exp": 1398927402,
      "client_id": "ACoolDesktopApp",
      "scope": "read",
      "AccountId": "137"
    }

### Are we encrypting our JWT Token?

No. We are not encrypting our JWT Token. We are encoding them. We are using a standard Microsoft library. A System.IdentityModel.Tokens.JwtSecurityToken Class.

This aligns with standard OAUTH2.

### More generally, what’s the difference between Signing and Encrypting?

When encrypting, you use their public key to write message and they use their private key to read it.

When signing, you use your private key to write a message’s signature, and they use your public key to check if it’s really yours.

We sign our JWT Token so we can be sure it was us who originally created it.

On a privacy point… Anyone who gets one of our JWT Tokens can decode it and easily read it’s contents. When they attempt to reencode it they will need to sign it with a signing key.

### Our Signing Key

Our signing key is important. It drives the basis for our authorisation. When we receive a Token we check if it’s signing key matches ours.

Our signing key is hosted by us. Clients don’t rely on the signing key.

Clients simply take a token and then include it in every request. We access that Token for authorisation.

In other words: * You create a header object, with the JSON format. Then you encode it as a base64 * You create a claims object, with the JSON format. Then you encode it in base64 * You create a signature for the URI (we’ll get into that later). Then you encode it in base64 * You concatenate the three items, with the “.” separator

### HS256

Our hashing Algorithm. By default we use the HS256 algorithm and our signing key to encode the token.

### SSL Encrypts Everything!

As per the standard. The only thing that ensures our approach remains secure is that we rely on SSL for every request. This prevents anyone from stealing someone else’s Token.

### Who’s Playing?

To simplify things…

*   A Mobile App Client.
*   A Authentication Web Api.
*   A Resource Web Api.
*   A WCF Service that shields our backend.

How do these players collaborate?

An mobile app client requests an token from the authentication web api and uses to request data from a resource web api. The resource web api takes that request, assesses the clients authorisation and then passes it onto a trusted WCF Service. The WCF Service currently does not authenticate the request. The WCF Service trusts anything the Resource Web Api feeds it.