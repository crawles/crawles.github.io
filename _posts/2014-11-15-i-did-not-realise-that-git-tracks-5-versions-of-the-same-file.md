---
layout: post
title: I did not realise that Git tracks 5 versions of the same file.
published: True
feature: 

---

*   [Git Fundamentals](http://www.pluralsight.com/courses/git-fundamentals)
*   [Aha!](http://betterexplained.com/articles/aha-moments-when-learning-git/)

I’m currently working on a project using Git without Pull Requests. It’s an old version. Hopefully we will upgrade soon.

If you can, I recommend working with Git & Pull Requests. It’s a whole world better. Better than vanilla Git. Better than TFS. It solves many of the hot issues that occur when dealing with opinions and version control. Github’s Pull Requests allow for a more proactive social collaboration approach to version control and opinion management.

It’s been interesting doing a little researching on vanilla Git.

## Git tracks 5 versions of the same file!?

Git hosts and maintains an independent copy of any given file in the following 5 areas. Git commands help manage the flow and progress from Working Copy to Remote. Each one of these areas can reflect conflicting versions and opinions.

- Working Copy
- Staging
- Local Repository
- Remote Repository (local cache)
- Remote Repository