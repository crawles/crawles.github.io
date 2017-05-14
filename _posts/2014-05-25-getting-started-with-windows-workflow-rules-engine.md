---
layout: post
title: GETTING STARTED WITH WINDOWS WORKFLOW RULES ENGINE
published: True
feature: 

---

*   [A quick and dirty Rules Engine using Windows Workflow](http://sanderstechnology.com/2011/a-quick-and-dirty-rules-engine-using-windows-workflow-part-2/10491/#.U4J1YPmSyCk) by Rob Saunders

Following on from an experimentation with NxBRE, here’s a sample review of using the Windows Worflow Rules Engine that ships as part of .Net 4x.

## System.Workflow namespace

.Net 4x ships with a namespace that provides a comprehensive list of Rules Engine functionality.

<table>

<thead>

<tr>

<th>Namespace</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>System.Workflow</td>

<td>The System.Workflow namespaces contain types used to develop applications that use Windows Workflow Foundation. These types provide design time and run-time support for rules and activities, to configure, control, host, and debug the workflow runtime engine.</td>

</tr>

<tr>

<td>System.Workflow.Activities</td>

<td>Provides classes related to Windows Workflow Foundation activities.</td>

</tr>

<tr>

<td>System.Workflow.Activites.Rules</td>

<td>Contains a set of classes that define the conditions and actions that form a rule.</td>

</tr>

</tbody>

</table>

### Workflow components

<table>

<thead>

<tr>

<th>Component</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>RuleEngine</td>

<td>Used to perform RuleSet evaluation.</td>

</tr>

<tr>

<td>RuleExecution</td>

<td>Stores state information while executing RuleCondition or RuleAction classes.</td>

</tr>

<tr>

<td>RuleValidation</td>

<td>Validates expression conditions.</td>

</tr>

<tr>

<td>RuleSet</td>

<td>Contains a collection of Rule classes along with the semantics for forward-chaining execution of those rules. A RuleSet can be executed directly in code or using the PolicyActivity activity.</td>

</tr>

</tbody>

</table>

### Example implementation

To get started here’s a simple test method that fires up a set of rules and executes these rules on the incoming object

    [TestMethod]
    public void CreateNewRule()
    {
        Employee testEmployee = new Employee();

        testEmployee.FirstName = "Joe";
        testEmployee.Surname = "Smith";
        testEmployee.Location = StateEnum.ACT;
        testEmployee.Manager = null;
        testEmployee.DateHired = DateTime.Now.AddYears(-1);
        testEmployee.EmployeeNumber = 99;

        string ruleName = String.Format("{0}UnitTestRule", DateTime.Now.Millisecond);
        string path = Assembly.GetExecutingAssembly().Location.Replace(Assembly.GetExecutingAssembly().ManifestModule.Name, String.Empty);

        RuleSet newRule = RulesManager<Employee>.LaunchNewRulesDialog(ruleName, path);
        testEmployee = RulesManager<Employee>.ProcessRule(testEmployee, newRule);

        Trace.WriteLine(testEmployee.FirstName);
        Trace.WriteLine(testEmployee.Surname);
        Trace.WriteLine(testEmployee.DateHired);
    }

The ProcessRule looks like this…

    /// <summary>
    /// Execute a single rule on a single data object
    /// </summary>
    /// <param name="objectToProcess"></param>
    /// <param name="rule"></param>
    /// <returns></returns>
    public static T ProcessRule(T objectToProcess, RuleSet rule)
    {
        RuleValidation validation = new RuleValidation(typeof(T), null);
        RuleExecution execution = new RuleExecution(validation, objectToProcess);
        rule.Execute(execution);
        return objectToProcess;
    }

This approach takes certain components form the Windows.Workflow namespace that ships with .Net 4x.