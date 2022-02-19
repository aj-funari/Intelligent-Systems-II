# lab9

Packaging your ML application:  Proper use of pip freeze, application transferability, application directory structure, testing the installation of your application and project examples.

# Connecting a Docker container to a local dir

In order to make Docker a full blown development environment we need to have the ability to save our work so it exists when the container is no longer running. Here we will showcase how to create a local directory and map it to the container so that you can save things to that directory from the container and from the local volume. Think of this as a two way street. 

## Introduce the -v flag

```
-v /your/path/to/the/dir/you/created/locally:/working_dir_in_the_container
```

The -v is what creates the bridge between the container and your local storage space, in this case: /your/path/to/the/dir/you/created/locally , the /working_dir_in_the_container is most easily set from within the Dockerfile like so:

```
WORKDIR working_dir_in_the_container/
```
For clarity please use dir names that are suitable for you, I have created these for instruction purposes only. 

## Review the -i and -t flags 

Remember the -i tag allows us to use the container interactively and the -t allows us to name the container so we can use the tag name versus the Docker naming convention, which is much longer. 

The following two commands will; first build you container and give it a name; then it will run the contianer interactively. 

```
docker build --pull -t pkg_demo .
```

```
docker run -p 8080:8080 -v /your/path/to/the/dir/you/created/locally:/working_dir_in_the_container -it pkg_demo
```

# Python packaging (minimal example)
```
top_level (usually a repo i.e. top_level.git)
  |- module_name (this is the actual python module)
  | |- __init__.py
  | |- text.py
  | |- sum.py
  |- setup.py
  |- .gitignore
```

Once we have this directory structure we can go to the top_level and use pip to install locally. 

```
cd ~/top_level
pip install .
>>> import module_name
>>> print(module_name.text())
```


