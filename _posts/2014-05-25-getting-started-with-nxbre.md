---
layout: post
title: GETTING STARTED WITH NxBRE
published: True
feature: 

---

*   [NxBRE Project Files](http://sourceforge.net/projects/nxbre/)
*   [NxBRE Github Repository](https://github.com/ddossot/NxBRE)
*   [Basic Help on NxBRE (Rule Engine)](http://www.codeproject.com/Articles/404367/Basic-Help-on-NxBRE-Rule-Engine)
*   [NxBRE Documentation](http://www.docstoc.com/docs/42784923/NxBRE-Documentation-3_2_0)

## What is a Business Rules Engine?

I’ve spent the past few weeks exploring what a business rules engine is and what it isn’t.

Interestingly. A business rules engine doesn’t actually do anything. It helps handle complex questions. But it doesn’t actually help carry out actions.

A Business Rule is made up of the following broad concepts.

<table>

<thead>

<tr>

<th>Concept</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Rule</td>

<td>A complex business decision with a certain outcome.</td>

</tr>

<tr>

<td>Condition</td>

<td>A single piece of logic that is being tested.</td>

</tr>

<tr>

<td>Outcome</td>

<td>The result of the Rule Condition. E.g. The outcome could potentially be Pass or Fail.</td>

</tr>

<tr>

<td>Action</td>

<td>Actually doing something. Persisting a change. Modifying something. Doing something as a result of the rule.</td>

</tr>

</tbody>

</table>

These terms are slightly contentious. In general, in normal speaking english. When we talk about a Business Rule we refer to the combination of a Condition, and Outcome and an Action. Right?

## Getting the NxBRE Project

1.  Grab the [Project Files](http://sourceforge.net/projects/nxbre/) or Fork the [Github Repository](https://github.com/ddossot/NxBRE).
2.  Fire up the solution in Visual Studio.
3.  Resolve NUnit Dependencies and build.

## An Example Project

To experiment with the framework kick off the Examples Solution file that you’ll find in the ~/Examples folder.

To test run the Discount Project take a quick look at the Discount.xbre file in the ~/Rulefiles folder. You’ll need that to run the console app.

    > Discount.exe ../../../../RuleFiles/discount.xbre

## XBRE Files

An XBRE file is an Business Rules Engine definition written in XML.

    <xBusinessRules>
        <!-- global values -->
        <Integer id="10i" value="10"/>
        <ObjectLookup id="QuantityOrdered" objectId="CurrentOrder" member="Quantity"/>
        <Logic>
            <If>
                <Do>
                    <!-- Discount rules for high rate customers -->
                    <Evaluate id="AppliedDiscount">
                        <Parameter name="Percent" value=".7"/>
                    </Evaluate>
                </Do>
            </If>
            <Else>
                <!-- Discount rules for low rate customers -->
                <And>
                    <GreaterThan leftId="QuantityOrdered" rightId="10i"/>
                </And>
                <Do>
                    <Evaluate id="AppliedDiscount">
                        <Parameter name="Percent" value=".8"/>
                    </Evaluate>
                </Do>
            </Else>
        </Logic>
    </xBusinessRules>

It’s made up of the following key entities.

<table>

<thead>

<tr>

<th>Entity</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>Do</td>

<td>Carry out an action.</td>

</tr>

<tr>

<td>Evaluate</td>

<td>Calculate a condition for an outcome.</td>

</tr>

<tr>

<td>ObjectLookup</td>

<td>Reflection call on a class or an object. Reads the value of a member of a particular object (referenced by objectId) and stores its value in the context under the given id. Performs a call to a static member of an helper class, providing arguments for the call, and storing the result in the context under the given id. Calls a particular member of a particular object (referenced by objectId), providing arguments for the call. The engine’s context is not modified. Calls the .NET delegate that have been bound under id in the engine context prior to processing the rule base.</td>

</tr>

<tr>

<td>Logic</td>

<td>A logic block should contain a main condition (IF) and can contain an alternative (ELSE).</td>

</tr>

<tr>

<td>While</td>

<td>A while block first contains a conditional top element (AND, OR, NOT) then the action element.</td>

</tr>

<tr>

<td>ForEach</td>

<td>Enumerate the object stored in the context under valueId and place its current value under id.</td>

</tr>

<tr>

<td>Set</td>

<td>Defines a set which is a group of any operations (except Set) identified by an id.</td>

</tr>

<tr>

<td>ThrowException</td>

<td>Posts an empty Error event to the FlowEngineRuleBase trace source.</td>

</tr>

<tr>

<td>ThrowFatalException</td>

<td>Posts a Critical event with the provided text to the FlowEngineRuleBase trace source.</td>

</tr>

<tr>

<td>Log</td>

<td>Posts an event with the provided text to the FlowEngineRuleBase trace source. The level is used as a numeric index in the TraceEventType enumeration. If no matching type is found, Information is used.</td>

</tr>

</tbody>

</table>

## Loading our rules.

Here’s the first step in loading the xml and fire up a BREImpl object. A Business Rules Engine Implementation object.

    public CalculateTotal(string aXMLFile, SourceLevels engineTraceLevel, SourceLevels ruleBaseTraceLevel)
    {
        bre = new BREFactoryConsole(engineTraceLevel, ruleBaseTraceLevel).NewBRE(new XBusinessRulesFileDriver(aXMLFile));
        if (bre == null) throw new System.Exception("BRE Not Properly Initialized!");

        bre.RuleContext.SetFactory(APPLIED_DISCOUNT, new BRERuleFactory(new ExecuteRuleDelegate(AppliedDiscount)));

        // example on how to export native rules file
        FileStream fs = new FileStream("discount.bre", FileMode.Create);
        Xml.IdentityXSLT.Transform(bre.XmlDocumentRules, null, fs);
        fs.Flush();
        fs.Close();
    }

## Using our rules.

Once we have our rules loaded we can use them!

    bre.RuleContext.SetObject(ORDER, aOrder);
    bre.Process();

## NxBRE DSL Language

The whole point of using a Rule Engine is to help Business Owners get in on the conversation. To help with transparency, testability, re-usability etc…

Once we get our rules up and running the next step is to broaden the self documenting nature of a rule by using Domain Specific Language.

    rule "Discount for regular products"
    if
    The customer is rated premium
    and
    The product is a regular product
    then deduct
    The discount on this product for the customer is 5.0 %