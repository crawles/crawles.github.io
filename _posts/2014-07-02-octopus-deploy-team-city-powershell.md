---
layout: post
title: OCTOPUS DEPLOY, TEAM CITY & POWERSHELL
published: True
feature: 

---

20 minutes with Powershell and I’ve got myself a running multi environment continuous deployment management setup running off octopus deploy and team city. Love it. 5 environments. Many more machines. Now, auto managed.

### More reliable release to production

Pain point. Octopus. Automated deployment tool. Take a .net app and release to various environment.

### Changing configurations. Easy

Mainly for asp.net web applications, databases, and windows services. Avoid desktop apps and mobile apps.

### Really easy to use. Highly visible

Replaces the need for rdp. Old approaches end up breaking great software. People do still use rdp – madness.

Merged with tfs version control. History Visibilty. A clean connection with tfs tasks and bugs. Stories. Progress.

### Other relevant tools

Cruise control. Draco. DBUP.

Unit tests. Build.

Hidden rdp deployment. Not traceable.

The issue of the forgot deployment process. The scary out of date word doc.

### History

Powershell. Octopus is massively dependant on powershell. It’s everywhere ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

Team city builds the code / artifacts. How to get to the target machine?

Setting up websites from scratch. Bindings. Ports. Authentication. Starting. Creation. Application pools. Running things in the right context.

Powershell scripts formation. Reams of Powershell. Reduce the amount of Powershell. Breaks up scripts for each step. Tentacles. Agents.

Bypassing permissions across the wire. Octopus skips these issues. Trusted hosts and Powershell remoting gets difficult.

Octopus tentacles create a trust relationship with an x509 certificate. From there things get easy. Across an encrypted connection.

DEV. QA. UAT. PRE PROD. PROD. the process should be the same. Changing the release process introduces massive chaos.

### Orchestration

Once deployment is easy the safety increases and regular. deployment becomes more regular and successful. Trying to remember this stuff is simple unprofessional.

Aims to reduce Powershell but there’s no way to avoid it completely.

Question. What about setting different ports for a given process on different machines?

### Database deployment is tricky

Complex organisational issues. Out solution is sequential scripts that build on top of one another tha align with our sprint history.

This can get tricky and cumbersome in various testing and uat environments. At the moment we have a few manual rdp work around for dealing with this.

Manual pause step??? Great for bits that allow for not automated stuff. Great for a review.

### Angular and REST

Octopus Deploy is built with a regular Angular.JS web interface and a Nancy based REST api. This api can be accessed from any sort of client.