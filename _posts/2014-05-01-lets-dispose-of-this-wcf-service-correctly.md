---
layout: post
title: LET'S DISPOSE OF THIS WCF SERVICE CORRECTLY
published: True
feature: 

---

*   [Understanding Lifetime Managers][1]
*   [Microsoft.Practices.Unity Namespace][2]

Currently sorting through a line of code at work that is confusing me. It’s throwing an error.

    private DisposableHelper _channelDisposer = new DisposableHelper();
    ...
    _channelDisposer.Using(_authService);

## What is a DisposableHelper? Who cares?

Hmm… Interesting. I can see it’s got something to do with an IDisposable Interface and a design pattern I’ve never heard of before. The Disposable design pattern.

### IDisposable

Provides a mechanism for releasing unmanaged resources.

What? unmanaged? Are we in the CLR? Isn’t that something we don’t need to worry about?

    "Implement IDisposable only if you are using unmanaged resources directly."

Ah. This is starting to make a little more sense.

    "All programs acquire one or more system resources, such as memory, system handles, or database connections, during the course of their execution. Developers have to be careful when using such system resources, because they must be released after they have been acquired and used. A great example of this is the System.IO.Stream class. Although it is an abstract base class that doesn't hold any resources, most of its subclasses do and because of this, it implements this pattern."

The WCF Service resource we are depending on may be an unmanaged resource. Hmm. Given the depending WCF Service lives outside the currently CLR I guess it’s not something that is automatically handled by garbage collection.

## What is the Dispose Design Pattern?

Sorry. I know. This is dry. This is exactly the sort of thing you’ll miss when you’re next unemployed.

    "The CLR only controls and manages its memory."

Oh…. Here’s a basic implementation.

    public class DisposableResourceHolder : IDisposable {

        private SafeHandle resource; // handle to a resource

        public DisposableResourceHolder(){
            this.resource = ... // allocates the resource
        }

        public void Dispose(){
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // The Boolean parameter disposing indicates whether the method was invoked from the IDisposable.Dispose implementation or from the finalizer.
        protected virtual void Dispose(bool disposing){
            if (disposing){
                if (resource!= null) resource.Dispose();
            }
        }
    }

And now it’s becoming very clear that this design pattern is merely a way of wrapping an object. The basic implementation of the pattern involves implementing the System.IDisposable interface and declaring the Dispose(bool) method that implements all resource cleanup logic to be shared between the Dispose method.

## Finalizer Who? What?

    The CLR provides some help in releasing unmanaged resources. System.Object declares a virtual method Finalize (also called the finalizer) that is called by the GC before the object’s memory is reclaimed by the GC and can be overridden to release unmanaged resources. Types that override the finalizer are referred to as finalizable types.

Got it. This IDisposable object can be instantiated manually or left to be run by the CLR, when it’s good and ready.

## OK Great. I’ve got half and Idea. So Why?

Why is my Disposable object returning an error?

    System.ObjectDisposedException was unhandled by user code
      HResult=-2146232798
      Message=Cannot access a disposed object.
    Object name: 'System.ServiceModel.Channels.ServiceChannel'.
      Source=mscorlib
      ObjectName=System.ServiceModel.Channels.ServiceChannel
      StackTrace:
        Server stack trace: 
           ...
        Exception rethrown at [0]: 
            ...
      InnerException: 

Hmmm… Interesting. It appears the Dispose method has already been called on the object. Are we disposing of the function at the correct time?

## Got it!

The channel factory that is creating these WCF Services exists within a Singleton. When the controller finishes it’s task it Disposes of the service.

The issue was that the channels were being registered as singleton objects (scoped at the lifetime of the application or IIS process!!!). As soon I I made them close correctly at the end of each controller request, the issue presented itself.

Within our UnityConfig in the App.Common project, we were registering the IService instances like this:

    container.RegisterType<IAuthService>(new ContainerControlledLifetimeManager(),new InjectionFactory(
        (c) => new ChannelFactory<IAuthService>(binding, new EndpointAddress(baseUrl + "AuthService.svc")).CreateChannel()));

Our container’s are set up such that each web application (sso, api…) create a Unity container on ‘Application_Start’ so there is one container set up for the life of the running application. ContainerControllledLifetimeManager, as was being used above, means ‘Singleton for the Life of the Container’. So we were previously using only one instance of the service proxy’s for the lifetime of the app.

Unity’s default is actually TransientLifetimeManager() which means new instance every time the object is resolved.

What we actually want is something called HeirachycalLifeTimeManager, which means a singleton for the life of the child container. Which means, hold onto our underlying WCF Service for as long as the Web Api Controller has been instantiated. Given the Controller is reinstantiated on each request, our WCF Service resource will be released after each web api request.

We are taking this approach to ensure connections to our WCF layer are closed as quickly as possible.

## Lifetime Managers

### TransientLifetimeManager

This creates and returns a new instance of the requested type for each call to the Resolve or ResolveAll method. This lifetime manager is used by default for all types registered using the RegisterType, method unless you specify a different lifetime manager.

Typically this means that Every time the application starts (Start_App), the resource is kept alive.

### ContainerControlledLifetimeManager

A LifetimeManager that holds onto the instance given to it. When the ContainerControlledLifetimeManager is disposed, the instance is disposed with it.

### HierarchicalLifetimeManager

A special lifetime manager which works like ContainerControlledLifetimeManager, except that in the presence of child containers, each child gets it’s own instance of the object, instead of sharing one in the common parent.

### PerThreadLifetimeManager

A LifetimeManager that holds the instances given to it, keeping one instance per thread.

### ExternallyControlledLifetimeManager

A LifetimeManager that holds a weak reference to its managed instance.
