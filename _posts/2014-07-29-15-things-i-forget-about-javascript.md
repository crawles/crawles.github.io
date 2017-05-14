---
layout: post
title: 15 things I forget about JavaScript when I write C#
published: True
feature: 

---

*   [JavaScript for C# Developers](https://tv.labs.readify.net/videos/watch/mWtpOBzGaW4/javascript-for-c-developers)
*   [7 ways to create objects in Javascript](http://a-developer-life.blogspot.com.au/2011/11/7-ways-to-create-objects-in-javascript.html)
*   [HTTP Request & Response Service](http://httpbin.org)

> “If it quacks like a duck, but looks like a cat with a horse head, it must be JavaScript.” -@rodneyrehm

Now. I’m not new to Javascript or C#. I’ve been exploring both for years. But more recently I’ve found myself flipping between the two. I want them to be the same, but it’s just not going to happen.

### 1. Dictionaries

Javascript functions / objects are more like dictionaries.

    var a = [];
    a['Name'] = 'Kahne';
    console.log(a.Name); // ?! Array. Dictionary. Property? Array style. Dot notation.

### 2. Anonymous functions

What the heck is an anonymous function? It’s a function without a name.

    function a(){
        // this is a normal function. this gets called.
    }

    function (){
        // this is an anonymous function. this always runs.
    }

### 3. Functions are a first class entities

Functions are Objects. Functions are a first class entity. In C#, functions are 2nd class concepts, they must belong to objects.

    var foo = function(){
        // do function stuff.
    }

    var x;
    x.f = foo; // this variable is a function.

    x.f(); // can now be invoked.

### 4. Context

This. In javascript, functions are independant. So how can ‘this’ make any sense?

When a function runs it’s given a ‘context’.

What is the context? It’s the thing on the left of the dot.

    x.foo(); // if foo was a function passed into x, when it is run, 'this' will refer to the x context.

If you don’t specify a context from the first line – your line of code will run within the “window” context. This leads to messy code.

### 5. Closure. Confusion.

This is a sample that looks like it should print 42, but it actually prints out 0\. This is because the context in Javascript is something that requires additional awareness.

    var z = {
        i : 0, 
        fn : function(){
            var a = function(){
                this.i = 42; // set i in current context.
            };
            a(); // run "a()" in "window" context.
            console.log(i); // print "i" in "z" context.
        }
    };
    z.fn(); // 1\. Run a function in context "z";

This ‘context’ of a function is something that can easily change. It is defined by the caller.

### 6. Camel Case

With regular OO everything is Pascal case. PascalCase. In Javascript, everything is Camelcase. camelCase. Except for functions that represent an object. These guys are declared with a function in Pascal case.

Everything else is camel case. Including file names and folders etc…

    function Person(name, email){
        this.name;
        this.email;
        this.sendEmail = function(){
            // ...
        };
    };

    ...
    var person = new Person('John Smith', 'js@email.com');
    person.sendEmail();

### 7. The horrors of the window context

How do we get our objects and variables in a safe space, away from the global window context?

Use an iffe. An Immediately-Invoked Function Expression.

    // sample 1
    var iffe = ()();

    // sample 2
    var iffe = (function(){
        // ... independant scope.
    })();

    // sample 3
    (function(){
        // ... independant scope.
    })();

Once inside an iffy the way to create a complex namespace structure is to fake it. There is no such thing as a namespace concept in JavaScript. Instead, we need to create functions / objects and place functions / objects inside them.

    Customer = {};
    Customer.Order = {};
    Customer.Order.Invoice = new Invoice();

### 8. Interpreted language

JavaScript is an intepreted language. It’s not compiled. The code discovers functions and objects as it goes. Therefore. The order matters. If you call a function, it needs to be linked up.

Most of the time this shouldn’t really matter because of the way JavaScript thinks ahead with variable hoisting and function hoisting. When you instantiate a function, it hoists it’s internals so you don’t have to worry about the order in which they are loaded – when calling them from ‘outside’ the function.

JavaScript is a scripting language.

Javascript rearranges the internals of a given function so it will always load internal functions first.

### 9. This. That.

Why would we ever want to store ‘this’? Well. Because we want to grab a hold of the context when the object is initialised.

A solution to the above problem could be…

    var z = {
        i : 0, 
        fn : function(){
            var that = this;
            var a = function(){
                that.i = 42;
            };
            a();
            console.log(i);
        }
    };
    z.fn();

### 10. Endless parameters

What’s with the random listing of parameters in JavaScript?

    var fn (a, b)
    {
        // a == 123.
        // b == 567.
        // arguments[2] == 789.
    }

    fn(123,456,789); // this works.

[arguments] is a property that always there. The list is soft. This means it’s not possible to overload a function in Javascript.

### 11. Prototype -ing

Every function (object) has a common set of properties. One of these common properties is “Prototype”.

    function Person(name, email){
        this.name;
        this.email;
    };

    Person.prototype.sendEmail = function(){
        // ...
    };

    var person = new Person('John Smith', 'js@email.com');
    person.sendEmail();

Taking this approach means that the person object doesn’t need to create a new version of that function for each instantiation of the Person function. It’s more efficient.

Each new instantiation of Person now has a reference to the same function rather than it’s own copy.

Prototyping also gives us access to the ability to implement a sort of inheritance.

    function Person(){
        this.address = ;
    };

    function Customer(){};
    Customer.prototype = new Person();

    var c = new Customer

### 12. Instantiation of a new object

Yes. There is yet another way to create functions (objects).

    var person = {
        name : 'John Smith',
        email : 'js@email.com',
        sendEmail : function(){
            ...
        },
    };

    person.sendEmail();

This approach does everything in one go. It’s not great for reusability given the object cannot be instantiated with other values.

Yes. There is yet another way to create functions (objects).

    var person = new function(){
        this.name = 'John Smith',
        this.email = 'js@email.com',
        this.sendEmail = function(){
            ...
        },
    };

    person.sendEmail();    

This approach also does everything in one go. It’s not great for reusability given the object cannot be instantiated with other values.

### 13. Looping through an array

Firstly, arrays are weird. They can have undefined values. They can have any type of object for each individual item in the array.

There are two ways to run through an array.

    for(var i = 0; i < customers.length; i++){
        var c = customers[i];
    }

This does sort of return a different record set…

    for(var i in customers){
        var c = customers[i];
    }

What the differences of these two approaches shows is that, an array in JavaScript is just an object with a set of properties. Each item in the array is actually a property – with a value.

And another way to loop through an array. This won’t work in an old browser.

    customers.forEach(function(i){
        //...
    });

### 14. Truthy

    1 == 1 // true
    1 == '1' // true. ??? even though they are completely different.
    1 == 0 // false
    1 == true // true
    0 == true // false
    2 == true // false
    2 == false // false

Type coersion. Dangerous. Triple equals sets up a Direct Equality condition.

### 15. Loading scripts in a browser

Javascript runs in a browser. Everything gets loaded in from other files from everywhere and anywhere. How to get things loading quickly?

Should I be sticking my script tags in the head tag? No. Lame. That means nothing loads until the scripts are in. DELAYED. Crappy page load performance.

Stick script tags at the end of the page. CSS is at the bottom of the page.

    // FOUC: Flash of unstyled content. Pronounced "Fuck".