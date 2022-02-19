# Lab 4

With this lab we shift our focus from developing a machine learning model to creating a service that uses that model. There are two significant pieces needed: we'll define a specification file for the creation of the web api and a framework (tooling) that will help set up the handling of the requests and responses between the server and client. The latter framework has two modules: Flask and connexion.  Flask helps with instantiation of the web api.  Connexion handles http requests. Why these two? Simply, Flask is light weight and connexion conforms to OAS. What is OAS? Think of it as a set of clearly defined rules for the creation of a RESTful web api.  We illustrate through use of a toy python service that gets cpu information.

# Goal

1) Create a github account so we can clone this example. 

2) Use the Makefile to run the service. (Windows users may just have to use the docker commands in the makefile if make is not installed)

3) Identify within the .yaml file where the url is defined and

4) where the path to the python code is defined. 

5) Begin to understand how Flask, connexion and the .yaml file are interacting to create our service. 

6) add the get_os path to the .yaml file, it is located in the cpu_2020.py file. 

# Resources for you

## Read about OAS 

You are required to read about writing OAS definitions and the required reading can be found here:

[Defining OAS 3.0](https://swagger.io/docs/specification/basic-structure/)

[Defining OAS 2.0](https://swagger.io/docs/specification/2-0/basic-structure/)

This documentation walks you through the basics of your YAML file, we will be using YAML exculsively in this class, however if you want to use JSON for your project feel free. 

## connexion

To conform to OAS standards we will be using connexion to handle the HTTP requests. The documentation is found here:

[connexion docs](https://connexion.readthedocs.io/en/latest/)


# Start

`make docker-all`

now remember in order to stop the service use `make docker-stop` in a seperate terminal. This should work seamless. 

Look in os_pack there are two files. Looks at them. Redfine the operationid in the yaml file to point to the cpu_2020.py file. I have already created the def for the OS portion so the lab activity will be easy. 

Once the service is started and running you go here to test it out: [localhost:8080/engr-222/cpu](localhost:8080/engr-222/cpu)

# what you need to do for the lab activity

create a new endpoint that tells the user what OS they are running by defining a path and operationid in the yaml file.
