# ARKWater #

This repository contains a Java ML/NLP library used primarily with projects
associated with the Noah's ARK group (ark.cs.cmu.edu).  The library helps 
with two main tasks:  

1.  Running text documents through NLP pipelines 
(e.g. the Stanford CoreNLP pipeline 
(http://nlp.stanford.edu/software/corenlp.shtml)) to produce annotated
documents in a clean JSON format. 

2. Using the NLP annotated documents to construct featurized data sets 
that allow models to be trained and evaluated. 

Task 1 is completed by passing text into a class that extends 
ark.model.annotator.nlp.NLPAnnotator (e.g. ark.model.annotator.nlp.NLPAnnotatorStanford),
and using the resulting annotations to construct annotated-document objects that
extend ark.data.annotation.Document.  See 
temp.scratch.ConstructTempDocumentsTempEval2 in the TemporalOrdering
project (https://github.com/forkunited/TemporalOrdering) or 
textclass.scratch.ConstructTextClassDocuments20NewsGroups in the 
TextClassification project (https://github.com/forkunited/TextClassification)
for examples of this process.

Task 2 is completed by deserializing a set of NLP annotated documents
(ark.data.annotation.Document), constructing a data-set 
(ark.data.annotation.DataSet) containing datums (ark.data.annotation.Datum)
from the deserialized documents, and passing the data-set into
an experiment (ark.experiment.Experiment) that evaluates a model 
from ark.model on a featurized version of the data 
(ark.data.feature.FeaturizedDataSet) according to a process
specified in a class from ark.model.evaluation.  The experiments
from ark.experiment.Experiment are deserialized from experiment
configuration files that specify the model, features, and 
evaluations to use (see the 'experiments' directory in 
the TextClassification or TemporalOrdering projects mentioned above
for examples).  The deserialization process uses data utilities
from ark.data.DataTools and data-set specific factory methods
from in ark.data.Datum.Tools to construct model and feature objects
into which to load specific parameters.

Each project that uses ARKWater usually has a 'properties' configuration
file that contains machine-specific paths to data, experiments, etc.  These
files can be loaded into memory using classes that extend ark.util.Properties.
See temp.properties with temp.util.TempProperties or textclass.properties
with textclass.util.TextClassProperties for examples.

## Layout of the library ##

The code is organized into the following packages in the *src* directory 
(labeled with (1) and (2) to indicate which main task described above that the 
package is associated with):

*	*ark.data* (2) -  Classes for cleaning data and performing other miscellaneous 
data related tasks.

*	*ark.data.annotation* (1),(2) - Classes for loading NLP annotated documents and
data sets into memory.

*	*ark.data.annotation.nlp* (1) - Classes for representing various NLP annotations.

*	*ark.data.annotation.structure* (2) - Classes for organizing data into structures
for structured prediction models.

*	*ark.data.feature* (2) - Classes for featurizing sets of data so that they can
be used in models.

*	*ark.experiment* (2) - Classes for parsing and running experiments on models.

*	*ark.model* (2) - Implementations of various machine learning models.

*	*ark.model.annotator.nlp* (1) - NLP pipeline interfaces.

*	*ark.model.constraint* (2) - Classes representing constraints on data sets.  These
are currently only used by ark.model.SupervisedModelPartition to determine which
parts of the data should be used to train which models.

*	*ark.model.evaluation* (2) - Classes for carrying out various evaluation methods
like K-fold cross-validation

*	*ark.model.evaluation.metric* (2) - Evaluation measures (accuracy, F1, precision, 
recall, etc)

*	*ark.util* - Various classes and utilities for configuring projects,
 running external commands, dealing with files, dealing with Hadoop, etc.
 
*	*ark.wrapper* - Wrapper classes for external command-line utilities.

## How to build ##

You can use the build.xml included in the files directory to build using ant.
First, untar the required libraries in files/jars.tgz to the 
appropriate libraries directory, and then copy the build.xml to the root of the 
project.  Replace the text surrounded by square brackets in build.xml with
values that make sense for your setup.  Run  "ant build-jar" from the root of the project.

Alternatively, you can use an IDE like Eclipse instead of ant.

There is also currently a pre-built jar in the files directory.

## Possible Improvements ##

* The NLP annotated documents
should really be loaded into memory using ark.data.DocumentSet,
but currently this class is not finished, but there is something similar
to it in each project that uses ARKWater. 

* The deserialization for the experiment configuration files is
currently spread across several classes (e.g. ark.model.SupervisedModel,
ark.data.feature.Feature, etc) that all use ark.util.SerializationUtil.
This process is somewhat hacked together and should be made more intuitive.

* Create an ARKWater executable for carrying out typical generic tasks
like training various models or featurizing data sets.

* This documentation and the documentation at the top of each class file
is pretty limited, and the project is still probably difficult for someone
to understand without a good amount of effort.  So add more documentation.
