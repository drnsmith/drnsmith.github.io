---
date: 2023-03-20T10:58:08-04:00
description: "balbla bla blabla"
image: "/images/project1_images/pr1.jpg"
tags: ["network programming", "python"]
title: "BLA BLA BLA."
---

{{< figure src="/images/project1_images/pr1.jpg" caption="Image by Brett Sayles on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Client-Server-Network-Socket-Programming" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In the world of distributed systems, client-server architecture forms the backbone of communication between applications. 

This project demonstrates the development of a **secure client-server system** using Python, incorporating features like **serialisation**, **file encryption**, and **multi-threading** to ensure reliability and security.

This blog walks through the design, implementation, and key features of this system, offering insights into how these concepts can be applied in real-world scenarios.

### Project Overview

The project involved building a system where a client and server communicate securely to exchange serialised data and files. Key functionalities included:

1. **Data Serialisation**:  
Transforming Python objects (e.g., dictionaries) into transmittable formats.

2. **File Transfer**: Enabling clients to upload and download text files.

3. **Encryption and Security**: Protecting data during transfer with encryption and decryption.

4. **Multi-Threading**: Allowing the server to handle multiple client requests simultaneously.

### Setting Up Client-Server Communication

#### 1. Socket Programming Basics


