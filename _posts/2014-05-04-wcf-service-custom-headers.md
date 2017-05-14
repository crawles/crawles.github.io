---
layout: post
title: WCF SERVICE CUSTOM HEADERS
published: True
feature: 

---

*   [Adding custom message headers.](http://trycatch.me/adding-custom-message-headers-to-a-wcf-service-using-inspectors-behaviors/) By Eoin Campbell

WCF Services. Clients. Endpoints. Channels. Thinktecture. JWT. OAUTH. WebApi2\. Custom Headers. Unity. OperationContextScope. IDispatcherMessageInspector. IClientMessageInspector. AfterReceiveRequest.

Here’s a quick example on how to pass custom headers along with a regular WCF Service request.

[Grab the source code from my github repo here.](https://github.com/kahneraja/WCFCustomHeaderGarden)

This post might help answer some of these questions.

*   How do I pass a custom header from a client to a WCF Service?
*   How do I pass a Thinktecture JWT OAuth Token from a Web Api Application to a WCF Service?
*   How do I add a custom header to every message for a given service?
*   How do I send Typesafe values in a custom header?
*   How do I pass a users HttpContext to a WCF Service?
*   How do I pass a users ClaimsPrincipal in the header of a WCF Service message?

## Setup Your Garden.

To get started. Setup a basic WCF Service / Client solution.

This involves a hands on knowledge of a regular WCF Service and Client implementation. In Visual Studio, go ahead and create a WCF Service Application and a regular client application.

In the sample code provided I’ve created the following Projects. 1\. WCF Service Application. (Server) 2\. Web Api 2 Application. (Client) 3\. Common Library. (Service Contract, Data Contract & Extensions).

## ChannelFactory

    IHelloWorldService client = new ChannelFactory<IHelloWorldService>("BasicHttpBinding_IHelloWorldService").CreateChannel()

Start by setting up a basic Channel for you given service.

## Dependency Injection

Move your Channel into a IOC Container.

I’m making a special note on this because it adds a little complexity to the process of modifying the WCF Service header.

In my case I have many services. These services are utilised across a range of Web Api 2 applications. Each services exists with a Unity Container.

    container.RegisterType<IHelloWorldService>(
      new ContainerControlledLifetimeManager(),
      new InjectionFactory(
          (c) => new ChannelFactory<IHelloWorldService>("BasicHttpBinding_IHelloWorldService").CreateChannel()));

    container.RegisterType<IGoodbyeWorldService>(
      new ContainerControlledLifetimeManager(),
      new InjectionFactory(
          (c) => new ChannelFactory<IGoodbyeWorldService>("BasicHttpBinding_IGoodbyeWorldService").CreateChannel()));

## Message Inspection.

A message inspector is an extensibility object that can be used in the service model’s client runtime and dispatch runtime programmatically or through configuration and that can inspect and alter messages after they are received or before they are sent.

Client message inspectors implement the IClientMessageInspector interface and service message inspectors implement the IDispatchMessageInspector interface.

We want to get below the covers of the Service Contract. To do that we need to use a Message Inspector.

    public class ClaimsInspectorBehavior : Attribute, IDispatchMessageInspector, IClientMessageInspector, IEndpointBehavior, IServiceBehavior
    {
        #region IDispatchMessageInspector
        public object AfterReceiveRequest(ref Message request, IClientChannel channel, InstanceContext instanceContext)
        { ... }
        #endregion

        #region IClientMessageInspector
        public object BeforeSendRequest(ref Message request, IClientChannel channel)
        { 
            //Instantiate new HeaderObject with values from ClientContext;
            var dataToSend = new ClaimsHeader
            {
                Token = ClaimsHeaderContext.HeaderInformation.Token,
                AccountId = ClaimsHeaderContext.HeaderInformation.AccountId,
                RequestId = ClaimsHeaderContext.HeaderInformation.RequestId
            };

            var typedHeader = new MessageHeader<ClaimsHeader>(dataToSend);
            var untypedHeader = typedHeader.GetUntypedHeader("claims-header", "s");

            request.Headers.Add(untypedHeader);
            return null;
        }
        #endregion
    }

We now have the opportunity to add our messages to various outbound client requests in a single place.

## Custom Data Header Object

Create a type safe object that we can add to the header when we pass a message across the wire.

    [DataContract]
    public class ClaimsHeader
    {
        [DataMember]
        public string Token { get; set; }
        [DataMember]
        public int AccountId { get; set; }
        [DataMember]
        public int RequestId { get; set; }
    }

This is the object we’ll be using on both the client and the server.

## Client Context

The client application needs a single place to set some context values in one place so it doesn’t need to be passed around. An object to align in my case with the HttpContext and the Thread.Current.Identity.

    public static class ClaimsHeaderContext
    {
        public static ClaimsHeader HeaderInformation;

        static ClaimsHeaderContext()
        {
            HeaderInformation = new ClaimsHeader();
        }
    }

## Client Behavior

To modify the client behavior we need to implement IEndpointBehavior.

    public void ApplyDispatchBehavior(ServiceEndpoint endpoint, EndpointDispatcher endpointDispatcher)
    {
        var channelDispatcher = endpointDispatcher.ChannelDispatcher;
        if (channelDispatcher == null) return;
        foreach (var ed in channelDispatcher.Endpoints)
        {
            var inspector = new ClaimsInspectorBehavior();
            ed.DispatchRuntime.MessageInspectors.Add(inspector);
        }
    }

    public void ApplyClientBehavior(ServiceEndpoint endpoint, ClientRuntime clientRuntime)
    {
        var inspector = new ClaimsInspectorBehavior();
        clientRuntime.MessageInspectors.Add(inspector);
    }

## Server Behavior

To modify the client behavior we need to implement IServiceBehavior.

    public void ApplyDispatchBehavior(ServiceDescription serviceDescription, ServiceHostBase serviceHostBase)
    {
        foreach (ChannelDispatcher cDispatcher in serviceHostBase.ChannelDispatchers)
        {
            foreach (var eDispatcher in cDispatcher.Endpoints)
            {
                eDispatcher.DispatchRuntime.MessageInspectors.Add(new ClaimsInspectorBehavior());
            }
        }
    }

Update the service with this new behaviour attribute.

    [ClaimsInspectorBehavior]
    public class HelloWorldService : IHelloWorldService

## Configurable ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

Rather than programmatically setting the behaviour this can be done in configuration.

    <system.serviceModel>
    <behaviors>
      <endpointBehaviors>
        <behavior name="ClaimsInspectorBehavior">
          <ClaimsInspectorBehavior />
        </behavior>
      </endpointBehaviors>
    </behaviors>
    <extensions>
      <behaviorExtensions>
        <add name="ClaimsInspectorBehavior" type="HelloWorldCommon.Extensions.ClaimsInspectorBehaviorExtension, HelloWorldCommon" />
      </behaviorExtensions>
    </extensions>
    <bindings>
      <basicHttpBinding>
        <binding name="BasicHttpBinding_IHelloWorldService" />
        <binding name="BasicHttpBinding_IGoodbyeWorldService" />
      </basicHttpBinding>
    </bindings>
    <client>
      <endpoint address="http://localhost:59024/HelloWorldService.svc" 
                behaviorConfiguration="ClaimsInspectorBehavior"
                binding="basicHttpBinding" 
                bindingConfiguration="BasicHttpBinding_IHelloWorldService" 
                contract="HelloWorldCommon.ServiceContracts.IHelloWorldService" 
                name="BasicHttpBinding_IHelloWorldService" />
      <endpoint address="http://localhost:59024/GoodbyeWorldService.svc"
                behaviorConfiguration="ClaimsInspectorBehavior"
                binding="basicHttpBinding"
                bindingConfiguration="BasicHttpBinding_IGoodbyeWorldService"
                contract="HelloWorldCommon.ServiceContracts.IGoodbyeWorldService"
                name="BasicHttpBinding_IGoodbyeWorldService" />
    </client>
    </system.serviceModel>

## Call Your Extended Service

That’s pretty much it! You can now start running some tests and seeing if your custom header is ready to rock.

Set your context data somewhere in the given process…

    public HelloWorldController(IHelloWorldService helloWorldService, IGoodbyeWorldService goodbyeWorldService)
    {
        ...
        ClaimsHeaderContext.HeaderInformation.Token = Guid.NewGuid().ToString();
        ClaimsHeaderContext.HeaderInformation.AccountId = 23;
        ClaimsHeaderContext.HeaderInformation.RequestId = 135;
    }

And then in a given function call your WCF Service!

    public string Index()
    {
        var person = new Person() { FirstName = "John", LastName = "Smith", Age = 35 };
        var helloResponse = _HelloWorldService.SayHello(person);
        var goodbyeResponse = _GoodbyeWorldService.SayFarewell(person);
        return string.Format("{0}<br>{1}", helloResponse, goodbyeResponse);
    }   

## Read Your Custom Header

You should now have your fresh new custom header ready to use in your WCF Service. This custom header should show up for every method for the configured Services!

Encapsulate your custom header.

    [ClaimsInspectorBehavior]
    public class HelloWorldService : IHelloWorldService
    {
        public ClaimsHeader HeaderInformation
        {
            get
            {
                var claimsHeader =
                    OperationContext.Current.IncomingMessageProperties.FirstOrDefault(f => f.Key == "ClaimsHeader").Value as
                    ClaimsHeader;

                return claimsHeader;
            }
        }

        ...
    }

Call on it when you need it ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)

    [ClaimsInspectorBehavior]
    public class HelloWorldService : IHelloWorldService
    {
        public string SayHello(Person person)
        {
            return string.Format("Nice to meet you<br>{0}<br>{1}<br>{2}<br>{3}<br>{4}", person.FirstName, person.LastName, HeaderInformation.Token, HeaderInformation.AccountId, HeaderInformation.RequestId);
        }
    }

## Demo Results

When I run this HelloWorld Sample Project I get the following results from two separate WCF Services with a single client application.

    Nice to meet you
    John
    Smith
    a1c60054-aa8b-46c3-ab4d-4a50dffb9e65
    23
    135

    Farewell!
    John
    Smith
    a1c60054-aa8b-46c3-ab4d-4a50dffb9e65
    23
    135

In this example I’m passing simple data examples in our custom header. Alternatively I could use this ClientContext to pass the HttpContext, Http Headers, Identity, ClaimsPrinciple or any other client application contextual data the service might need. This approach works particular well if the client application and wcf service run within a trusted domain. If they trust one another they can share data more freely. This often occurs in flexible N-Tier Service Oriented Architecture solutions.