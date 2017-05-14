---
layout: post
title: THIS IS ME EXPLORING MARKDOWN
published: True
feature: 

---

*   [WP-Markdown By Stephen Harris](https://wordpress.org/plugins/wp-markdown/)
*   [Markdown Syntax By John Grube](http://daringfireball.net/projects/markdown/syntax)

# So what is Markdown?

I’ll repeat my answer to this again when I’m done. See the bottom of the page.

I’m interested in Markdown because I’m finding that I spend quite a bit of time writing. I check in code and then I write. I write technical documentation in a blogging style and I want it to look shmick.

## Mark down is…

A way to write technical documentation with references to ‘source code’ in the page. Right?

# Getting Started

Well. For now I’m with you on WordPress. So let’s see how we can shoe horn Markdown into WordPress.

# Shoe Horning Markdown Into WordPress

Yay! Google is my friend. WP-Markdown is a WordPress Plugin. This plugin Allows Markdown to be enabled in posts, comments and bbPress forums. This plug-in allows you to write posts (of any post type) using the Markdown syntax. The plug-in converts the Markdown into HTML prior to saving the post. When editing a post, the plug-in converts it back into Markdown syntax.

I’ll be right back. Just gotta go install and activate the WP-Markdown plugin under the covers.

Ok. Done. Now let’s see how this works.

# Getting the WP-Markdown plugin To Render Sexy Source Code.

Wait. Syntax? WP-Markdown Requires some form of Syntax. Let’s find out more about that.

# Markdown Syntax

Markdown Syntax is a human friendly syntax for generating HTML. It’s particularly useful for situations where you want to allow visitors to easily display code on a website.

Trying to render HTML using HTML is sort of like trying to spray water with a hose made out of water. As in, not possible.

To illustrate some Markdown Syntax I need to provide some examples.

## Wait. I’m now going Markdown Batshit Crazy

Something is going on in the background as I write this post. As I type I can see the resulting markup displayed below. I’m trying to figure out how to use Markdown to render Markdown examples. Yes. Water -> Water hose. Bare with me.

# Important Markdown Symbols

## Headings

### = Equals.

The “=” symbol is used to create Primary H1 Headings. Stick 3 in a row underneath some text and you’ll get a smoking heading.

### – Hyphen / Dash.

The “-” Hyphen symbol is used in the same was as the “=” symbol for creating Secondary H2 Headings.

### # Hash.

The “#” symbol is a more flexible way of creating headings. The more hashes you add the smaller the heading. “###” becomes a Sub-Sub H3 Heading. As you can see. I’m going to town with my Headings in this post.

## Phrasing

### * Star.

A “_” is used to *italics_ or _emphasis text_. The “_” symbol can be used to do the same thing.

## Listing

*, +, and – all produce the same thing. A list.

This

    * John
    * Jenny
    * Sally

Becomes

*   John
*   Jenny
*   Sally

### More Examples

    This is an [example link](http://example.com/).

### The Golden Rule For Escaping Into Code

To specify an entire block of pre-formatted code, indent every line of the block by 4 spaces or 1 tab.

    Yay!

Oh. And with an empty line above. Often an empty line above any new Markdown is required.

# So what is Markdown? Again.

    Markdown is a quick and easy way to organise technical documentation on the fly. I like it!

And now for a random unit test in Markdown just to celebrate.

    [TestMethod]
    public void Refresh_Token()
    {
        _StoredGrant.CreateRefreshToken = false;

        Claim[] claims = { new Claim("client_id", "MobileAppShop"), new Claim("secret", "12345678") };
        ClaimsPrincipal claimsPrinciple = Principal.Create("Test", claims);

        Thread.CurrentPrincipal = claimsPrinciple;

        TokenRequest tokenRequest = new TokenRequest()
        {
            Grant_Type = "refresh_token",
            Refresh_Token = "MyFavouriteRefrehToken1234"
        };

        var response = _TokenController.Post("Test Application 1", tokenRequest);
        TokenResponse tokenResponse;
        response.TryGetContentValue<TokenResponse>(out tokenResponse);

        Assert.IsTrue(response.IsSuccessStatusCode == true);
        Assert.IsFalse(string.IsNullOrEmpty(tokenResponse.AccessToken));
    }

This WP-Markdown plugin is great. It shows you the resulting content as your type it below the text editor. Nice one Stephen Harris ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)