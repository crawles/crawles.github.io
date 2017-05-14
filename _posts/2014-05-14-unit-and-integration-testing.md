---
layout: post
title: UNIT AND INTEGRATION TESTING WITH ENTITY FRAMEWORK AND UNITY
published: True
feature: 

---

*   [Isolating database data in integration tests](http://lostechies.com/jimmybogard/2012/10/18/isolating-database-data-in-integration-tests/)

Some quick notes on getting up and running with some unit tests and integration test.

### What is a unit test?

A unit test is an automated action that checks the functionality of a single unit of work. Or something…

This definition is causing everyone problems. Especially when you launch into the “Is TDD Dead?” debate.

How is it possible to test ANYTHING in isolation. I mean, isn’t everything dependant on everything else. E.g. Oxygen. Electricity.

Anyway. The point is. It’s very difficult to test a standard service oriented architecture solution. Design only gets you so far.

### Testing Controllers

To be honest. I’ve had to abandon this for now. Is that so bad? Perhaps the ‘testable’ logic shouldn’t be in the controllers anyway. Perhaps it should be encapsulated away into a business logic layer.

### Testing WCF Services.

This is where things get interesting. Firing up a WCF Service or Business Logic Manager Object within an integration test.

These internal objects exists within a Unity container.

And these objects all depend on an SQL Database. How to write a test that actually assesses the health of the baseline code and also reflects real world behaviour.

### Testing with a database, primary keys and foreign keys.

Rather than using a mocking implementation or adding a fake assembly I’ve found a very simple and quick way to get started.

Simple brace your integration test with the following transaction scope.

    [TestClass]
    public class CustomerIntegrationTests
    {

        private TransactionScope _TransactionScope;

        /// <summary>
        /// Ensures integration tests do not persist to the database.
        /// </summary>
        [TestInitialize]
        public void init()
        {
            _TransactionScope = new TransactionScope();
        }

        /// <summary>
        /// Ensures integration tests do not persist to the database.
        /// </summary>
        [TestCleanup]
        public void Cleanup()
        {
            _TransactionScope.Dispose();
        }

        /// <summary>
        /// Happy path.
        /// </summary>
        [TestMethod]
        public void AddCustomer()
        {
            var customer = new Customer()
            {
                Name = "Batman"
            };

            var newCustomer = CustomerManager.Add(customer );

            Assert.IsNotNull(newCustomer );
        }
    }

### Mocking.

Have you found that when you mock it ends up leading towards an endless rabbit whole of magic numbers? Well I have. It’s interesting to hear that many of the senior developers at Microsoft recommend NOT using any mocking.

### Speeding up the feedback cycle.

The great benefit of using this quick and easy solution is that you don’t have to fire up a local web server and manually test.

    Tip: To run any MS Test method click on the method signature and use the following keyboard command [CTR - R - T]!