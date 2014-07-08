Generally how experiment parsing works, featurization, model evaluation works
	- Datum tools, data tools
	- Should be improved later

Generally how document NLP annotation works
	- Should improve with DocumentSet later
	
Feature references
Parameter environment during deserialization
Datum extractor types used by features.

# ARKWater #

This repository contains miscellaneous Java utilities for some projects
associated with the ARK group (the ones worked on by Bill McDowell)--
namely, the OSI and Sloan projects.  

## Layout of the library ##

The code is organized into the following packages in the *src* directory:

*	*ark.data* - Classes representing data-structures for storing and/or
deserializing various types of data.

*	*ark.util* - Various classes and utilities for configuring projects,
 running external commands, dealing with files, dealing with Hadoop, etc.
 
*	*ark.wrapper* - Wrapper classes for external command-line utilities.

## How to build ##

You can use the build.xml included in the files directory to build using ant.
Copy that file to the root of the project, and then replace 

1.  Copy files/build.xml and files/corp.properties to the top-level directory
of the project. 

2.  Fill out the copied build.xml file with the appropriate settings by 
replacing the text surrounded by square brackets.

3.  Run the command "ant build-jar" from the root of the project.