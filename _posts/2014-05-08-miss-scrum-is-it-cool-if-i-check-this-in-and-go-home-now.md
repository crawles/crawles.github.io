---
layout: post
title: MISS SCRUM, IS IT COOL IF I CHECK THIS IN AND GO HOME NOW?
published: True
feature: 

---

I’ve been at my desk now for 4hrs, everyone has gone home, and things are getting a bit weird. And, I’m not under pressure from anyone but myself.

### Am I OCD or lazy?

I’ve been working with Visual Studio and Team Foundation Server now for nearly a decade. It’s home. Since going perm after a failed start-up concept (musomap.com) I’ve been working on in my spare time I’ve found there are some issues with collaborating. It’s a cultural thing. It’s an energy thing. It’s a space-to-fail thing.

“Do I check this in?” is a loaded question. The context is important. Different organisations have different policies.

I tend to check-in too often. Sorry.

Often when I sit with colleagues and review code we stumble upon unexpected concerns. I tend to get excited by this. I tend to think that “everything is my concern” and I shouldn’t say… “I’m not working on that so I’ll just let that slide.”

I tend to spend time on understanding the issue and I often step into resolving it. I consider the alternative to be careless and lazy.

### Step away from the concern!

Perhaps there is more to this. Perhaps it’s important to be more measured. Perhaps there are some major benefits to stumbling upon issues and realising their implications without grappling with the resolution.

One minute ago I was sat with a colleague writing code and he blew my mind. He taught me something.

We found this code.

    /// <summary>
    /// Save a new Request to the Database.
    /// </summary>
    /// <param name="resourceId"></param>
    /// <param name="accountId"></param>
    /// <returns></returns>
    public Request CreateRequest(int accountId, string endpoint)
    {
        ...
    }

There is nothing special about this code, except to us, it’s a big deal. The accountId is now obsolete.

We need to remove it. Quick!!!!

Given that we added it only a few days ago removing it, in my head right now, seems like a simple task.

1.  Remove weird unnecessary parameter from function.
2.  Build and see what happens.

My colleague didn’t make the change. He said something that reshaped my brain. He said…

“That’s not the task we are review right now. Let’s leave it.”

And in my head I thought..

“That’s lazy!”

### Fix it now in 1 minute or leave it for a day and it’ll take a month.

Often I bleed tasks because I feel close to them. The issue can reflect many different things.

*   Embarrassment. The issue reflect a lack of understanding on behalf and so I work to fix it asap.
*   Proximity. The issue sprung from a change that was recently made. The background knowledge is fresh in my mind. The symphony of connections are clear to me now, later they will be a mess.
*   Patterns. I get into a pattern and I want that pattern to appear everywhere. Repetition. Good software development is all about patterns right?
*   Testing. When I write something I want to make sure it works. To do that I need to see how it relates to other components. I want to link those components to see if it works. That often leads to bleed. I want to make sure it works by running alongside as many related components as possible. Integration work is messy.

### I lack focus.

All this chaos leads to a massive loss of focus. It makes writing code stressful. It’s confusing.

### Still at work at 7pm.

I care about what I write. I care about what people think of my code. I don’t leave work until my task is complete. But, given systems are like orchestras, how can one component every be done? It’s a issue I struggle with.

It all comes to a head when I attempt to check in my code. Is my work ready to be shared? Is my work ready for our shared development environment? Is my work done? Can I go home now?

Over the past many years I’ve found that I tend to stick around late and get stressed. My arm gets sore and my eyes get dry. It’s never really lead to a problem but it does affect my ability to enjoy the job.

### Did you break the build?

Yes. The dreaded banter, as you arrive at work the next day… “Hahaha you d1ckhead. You broke the build when you left last night. wtf?!”.

### My check-in is associated with my golden task.

My colleague, with his many years of experience, has taught me something today.

Be lazy. Focus on the task. When you see an unexpected issue that is outside the 1 or 2hrs you’ve got right now, pin it to the work board and palm it off out of your mind. Someone else will pick it up. Share the love!