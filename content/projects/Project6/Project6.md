---
date: 2023-03-20T10:58:08-04:00
description: "A deep dive into the implementation of a secure client-server application using Python socket programming. Learn how serialisation, file encryption, and multi-threading were used to create a robust communication system."
image: "/images/project6_images/pr6.jpg"
tags: ["network programming", "python"]
title: "Building a Secure Client-Server System with Python."
---

{{< figure src="/images/project6_images/pr6.jpg" caption="Image by Brett Sayles on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Client-Server-Network-Socket-Programming" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In the world of distributed systems, client-server architecture forms the backbone of communication between applications. This project demonstrates the development of a **secure client-server system** using Python, incorporating features like **serialisation**, **file encryption**, and **multi-threading** to ensure reliability and security. This blog walks through the design, implementation, and key features of this system, offering insights into how these concepts can be applied in real-world scenarios.

### Project Overview

The project involved building a system where a client and server communicate securely to exchange serialised data and files. Key functionalities included:

1. *Data Serialisation*: Transforming Python objects (e.g., dictionaries) into transmittable formats.

2. *File Transfer*: Enabling clients to upload and download text files.

3. *Encryption and Security*: Protecting data during transfer with encryption and decryption.

4. *Multi-Threading*: Allowing the server to handle multiple client requests simultaneously.

### Setting Up Client-Server Communication

#### 1. Socket Programming Basics

The server listens for incoming connections, while clients connect to send requests. Python's `socket` library was used to set up this communication.

#### Python Code: Server
```python
import socket
import threading

# Initialise server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8080))
server_socket.listen(5)
print("Server is listening on port 8080")

def handle_client(client_socket):
    """
    Handles client requests in a separate thread.
    """
    request = client_socket.recv(1024).decode()
    print(f"Received: {request}")
    client_socket.send("Acknowledged".encode())
    client_socket.close()

while True:
    client, address = server_socket.accept()
    print(f"Accepted connection from {address}")
    client_thread = threading.Thread(target=handle_client, args=(client,))
    client_thread.start()
```
#### Python Code: Client
```python
import socket

# Connect to server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", 8080))
client_socket.send("Hello, Server!".encode())

# Receive server response
response = client_socket.recv(1024).decode()
print(f"Server responded: {response}")
client_socket.close()
```
#### 2. Data Serialisation

To send complex objects (e.g., dictionaries), the system serialised data using Python’s pickle module.

#### Python Code: Serialisation
```python
import pickle

# Data to serialise
data = {"filename": "example.txt", "action": "upload"}

# Serialise and send
serialized_data = pickle.dumps(data)
client_socket.send(serialized_data)

# Server receives and deserialises
received_data = pickle.loads(client_socket.recv(1024))
print(f"Received: {received_data}")
```

#### 3. File Transfer

The system supported uploading and downloading files. Files were read in binary mode and transmitted in chunks for efficiency.

#### Python Code: File Transfer

```python

# Client: Upload file
with open("upload.txt", "rb") as file:
    while chunk := file.read(1024):
        client_socket.send(chunk)

# Server: Receive and save file
with open("received.txt", "wb") as file:
    while chunk := client_socket.recv(1024):
        file.write(chunk)
```

#### 4. Encryption and Decryption

To secure file transfers, encryption was implemented using Python’s cryptography library.

#### Python Code: Encryption
```python

from cryptography.fernet import Fernet

# Generate encryption key
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt file
with open("upload.txt", "rb") as file:
    encrypted_data = cipher.encrypt(file.read())

# Decrypt file
decrypted_data = cipher.decrypt(encrypted_data)
with open("decrypted.txt", "wb") as file:
    file.write(decrypted_data)
```

#### 5. Multi-Threaded Server

To handle multiple clients simultaneously, multi-threading was implemented. Each client connection was processed in a separate thread.

#### Python Code: Multi-Threaded Server
```python

import threading

def handle_client(client_socket):
    """
    Processes client requests in a separate thread.
    """
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print(f"Received: {data.decode()}")
            client_socket.send("Acknowledged".encode())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

# Accept multiple clients
while True:
    client, address = server_socket.accept()
    thread = threading.Thread(target=handle_client, args=(client,))
    thread.start()
```

### Challenges, Solutions and Applications
 - *Data Loss During Transfer*. Issue: Large files caused data to be dropped. *Solution*: Implemented chunk-based file transfer.

 - *Encryption Overhead*. Issue: Encryption slowed down the transfer speed for large files.*Solution*: Optimised encryption with asynchronous I/O.

 - *Concurrent Connections*: Issue: Multi-threading led to race conditions. *Solution*: Used thread-safe data structures.

This system has applications in:

 - *Secure File Transfers*: Transmitting sensitive files over networks.
 - *Real-Time Communication*: Chat or messaging systems.
 - *Data Serialisation*: Sharing structured data between systems.

### Future Directions and Real-World Applications

 - *Multi-Client Support*: Extending to a fully functional multi-client architecture.
 - *Scalability*: Integrating load balancers for handling high traffic.
 - *Improved Encryption*: Using hybrid encryption techniques for enhanced security.

The client-server system implemented in this project has versatile applications in various domains:

1. *Secure File Transfer Systems*: The system can be adapted for secure exchange of sensitive files, such as legal documents or medical records, over a network. *Use Case*: Hospitals transmitting patient records between departments.

2. *Real-Time Communication Platforms*: With modifications, the architecture can be used as a foundation for chat applications, collaborative tools, or notification systems. *Use Case*: Messaging apps for internal business communication.

3. *Remote Data Sharing*: Organisations can leverage the system to serialise and exchange structured data (e.g., JSON, dictionaries) between remote locations. *Use Case*: IoT devices sending telemetry data to centralised servers.

4. *Encryption for Security:* The encryption module ensures secure transmission, protecting against data breaches and unauthorised access. *Use Case*: Financial institutions exchanging transaction details between servers.

5. *Distributed Systems Development:* This project provides the groundwork for building distributed systems with multi-client and multi-server architectures. *Use Case*: Scalable cloud services for data processing.

### Conclusion

This project demonstrated the design and implementation of a secure client-server communication system. By combining socket programming, serialisation, encryption, and multi-threading, we created a robust and scalable architecture. The lessons learned here can be extended to build more advanced and secure distributed systems.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*

