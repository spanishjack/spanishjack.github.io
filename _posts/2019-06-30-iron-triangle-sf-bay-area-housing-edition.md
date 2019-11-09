---
layout: post
title: Iron Triangle - SF Bay Area Housing Edition
tags:
  - analytics
  - python
  - data
  - housing
  - analysis
---

![Iron Triangle - SF Bay Area Housing Edition]({{ site.url }}/images/IronTriangleSF.png)


# Overview
Are you shopping for a home in the SF Bay Area? Looking to move to the area for your next gig? If you've started your home search, your likely equal parts shocked, disgusted, and disallusioned. Don't worry. Your not alone. This post will give you a framework and data to help with your decision making process.

# Background
Living in the SF Bay Area is all about trade-offs. Meeting your lifestyle goals will mean navigating several constraints. Want a short commute and access to high performing schools for your children? Get ready to pay an exhorbiant amount for housing costs. Looking to save money on housing? Prepare to sign up for a brutually long commute. You can't have it all.

# Iron Triangle

*"Cheap, fast, or quality. Pick two."* 

Known as the [project management triangle](https://en.wikipedia.org/wiki/Project_management_triangle), Triple Constraint, or Iron Triangle. Maximizing your lifestyle in the Bay Area is a simliar constraint problem.

For many, the key factors in the decision making process are: 
 - Distance to job centers (commute)
 - Cost of housing
 - Quality of local schools

This gives us The Iron Triangle - Bay Area Living Edition. **Short commmute, affordable housing, or quality schools. Pick two.**

![Data Pipeline]({{ site.url }}/images/HousingTriangle.png)


Given this framework, your options will be:
 - Short commute + quality schools = expensive housing
 - Short commute + affordable housing = poor performing schools
 - Quality schools + affordable housing = long commute
 - Quality schools + affordable housing + short commute = lol! Not happening

In this model, the price of a home is a function of: 1) the distance to job centers (commute) 2) quality of local schools.

Home price = f(distince to job centers, quality of schools)

This gives us the following relationships:
  1. Job center distance vs home price: For a given home, holding school performance equal, the price of the home will be higher if the commute is shorter. Conversely home prices will be lower for homes with longer commutes.
  2. School performance: Given two homes with equal commutes, the home with the higher quality school will cost more.

![Data Pipeline]({{ site.url }}/images/PriceCommuteRelationship.png)


# Data Points for SF Bay Area
So, how close does reality line up to the model? Pretty close.

![Data Pipeline]({{ site.url }}/images/SFBayAreaScatterPlot.png)

The chart above shows zip codes plotted by:
  - Median price in USD (y-axis)
  - Distance to job center in miles (x-axis)
  - Color coded by high school performance ratings for the zipcode (quartile breakout)

As expected, the data shows a negative relationship between home prices and commute distances. And a positive relationship between home price and school performance.



Click [here]

<style>
.responsive-wrap iframe{ max-width: 100%;}
</style>
<div class="responsive-wrap">
<!-- this is the embed code provided by Google -->
<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSAjQGgootLFsZXPuI4DgzeYMznzwSQr69Sqkqm7Z2DSob0BNXH7_t3qhYmPpht705UOvYDFOBZGv4N/pubhtml?gid=1202104550&amp;single=true&amp;widget=true&amp;headers=false" frameborder="0" width="960" height="300" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
<!-- Google embed ends -->
</div>

<style>
.responsive-wrap iframe{ max-width: 100%;}
</style>
<div class="responsive-wrap">
<!-- this is the embed code provided by Google -->
<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSAjQGgootLFsZXPuI4DgzeYMznzwSQr69Sqkqm7Z2DSob0BNXH7_t3qhYmPpht705UOvYDFOBZGv4N/pubchart?oid=1368846724&amp;format=interactive" frameborder="0" width="960" height="300" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
<!-- Google embed ends -->
</div>