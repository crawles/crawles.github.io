---
layout: post
title: BACK TO WCF BASICS
published: True
feature: 

---

Windows Communication Foundation. A framework that brings together the long story of ASMX / WSE, .Net Remoting, DCOM/COM+/ES and MSMQ.

In a simpler WebApi OWIN world this technology is sort of being left behind. But I think it’s important to keep it in mind.

WCF is Microsofts major provider of service oriented programming model. Or, at least, it used to be and lots of larger organisations are still choosing to use it.

### What’s a Node?

The players. The various servers and clients involved. Each distributed component is a node.

### Transport Mechanisms

A transport mechanism describes the sort of data and the way the data is read by services and clients.

Various communication styles include; SOAP on MSMQ using XML, Distributed Objects on TCP using Binary and Restful on Http using Rss or Json.

### Endpoints

Each end point defines a different communication option. One service can have many endpoints that define a different transport mechanism. With configuration!

#### Addresses

A network address. where to send the message. An address that indicates where the endpoint can be found.

#### Bindings

How to send messages. ws* protocols. A binding that specifies how a client can communicate with the endpoint.

A recipe for how WCF will build the underlying communication channel.

#### Contacts

A contract identifies the operations available.

#### Behaviours

A set of behaviors that specify local implementation details of the endpoint.

### WCF Attributes

#### DataContracts

A data contract is a formal agreement between a service and a client that abstractly describes the data to be exchanged.

#### Data Members

Specifies that the member is part of a data contract.

#### ServiceContract

The service contract specifies what operations the service supports.

#### OperationContract

Indicates that a method defines an operation that is part of a service contract.

### Files & References.

#### System.ServiceModel.dll

ServiceContract & OperationContract WCF Interface components.

#### System.Runtime.Serialization.dll

DataContract & DataMember object serialisation.

#### App.config

Configure what services are being hosted and the available endpoints for each service.

### What’s the difference between WSHttpBindings and BasicHttpBindings?

If we want to summarize in one sentence, the difference between WsHttpBinding and BasicHttpBinding is that WsHttpBinding supports WS-* specification. WS-* specifications are nothing but standards to extend web service capabilities.

### An example WSHttpBinding message.

    <s:Envelope xmlns:a="http://www.w3.org/2005/08/addressing" xmlns:s="http://www.w3.org/2003/05/soap-envelope">
      <s:Header>
        <a:Action s:mustUnderstand="1">http://tempuri.org/IHelloWorld/SayHello</a:Action>
        <a:MessageID>urn:uuid:5a7f760d-e4a9-485a-a71a-4ab9bb466ca7</a:MessageID>
        <a:ReplyTo>
          <a:Address>http://www.w3.org/2005/08/addressing/anonymous</a:Address>
        </a:ReplyTo>
      </s:Header>
      <s:Body>
        <SayHello xmlns="http://tempuri.org/">
          <name xmlns:d4p1="http://schemas.datacontract.org/2004/07/HelloWorldService" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <d4p1:First i:nil="true" />
            <d4p1:Last i:nil="true" />
          </name>
        </SayHello>
      </s:Body>
    </s:Envelope>

### An example BasicHttpBinding message.

    <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
      <s:Header>
        <Action s:mustUnderstand="1" xmlns="http://schemas.microsoft.com/ws/2005/05/addressing/none">http://tempuri.org/IHelloWorld/SayHello</Action>
      </s:Header>
      <s:Body>
        <SayHello xmlns="http://tempuri.org/">
          <name xmlns:d4p1="http://schemas.datacontract.org/2004/07/HelloWorldService" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <d4p1:First i:nil="true" />
            <d4p1:Last i:nil="true" />
          </name>
        </SayHello>
      </s:Body>
    </s:Envelope>

### WebHttpBinding

RESTful via HTTP.

### BasicHttpBinding

SOAP via HTTP and limited WS* protocols.

### WSHttpBinding

SOAP via HTTP with full SOAP and WS* compliance.

### NetTcpBinding

Cross-machine communication via TCP. Binary across the wire.

### NetPeerTcpBinding

Cross-machine communication via P2P. Binary across the wire.

### NetNamedPipesBinding

Cross-machine communication via IPC. Binary across the wire.

### NetMsmqBinding

Disconnected Asyncronous communication via MSMQ. Binary across the wire.

### How does a client create a channel?

1.  Find a service.
2.  Retrieve the endpoint information. Via a WSDL file.
3.  Import meta data with references to available endpoints.
4.  Create a channel for a given endpoint.

Now, with that channel the client can now submit a message to the service.

### Setting the WCF Test Client

Interesting. By default the prebuild WCF Test Client application works great. To use a homegrown console app just update the startup options to link to your console app instead! Hit F5 of your wcf service and watch you exe file kick off ![:)](http://www.kahneraja.com/wp-includes/images/smilies/simple-smile.png)