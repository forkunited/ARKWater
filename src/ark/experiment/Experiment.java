/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of theMess (https://github.com/forkunited/theMess)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy 
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations 
 * under the License.
 */

package ark.experiment;

import java.io.BufferedReader;
import java.io.IOException;

import ark.data.annotation.Datum;
import ark.util.FileUtil;
import ark.util.SerializationUtil;

/**
 * Experiment represents an experiment usually involving a model and
 * a data set, specified through an experiment configuration file.  
 * The syntax of the input configuration
 * files is somewhat intuitive, so you can probably figure it out
 * by reading through the example experiments (for example in the 
 * experiments directory of the TemporalOrdering project
 * at https://github.com/forkunited/TemporalOrdering).  Generally,
 * each line of configuration is of the form:
 * 
 * [entityType]_[entityName]=[value]
 * 
 * Where the "_[entityName]" is optional.  For example, almost all experiments
 * of any type should have lines:
 * 
 * maxThreads=[maximum number of threads]
 * randomSeed=[random number generator seed]
 * 
 * Where the expressions right of the parentheses are replaced by numbers.  The
 * values on the right side of the equals sign can more generally be much
 * more complicated (for example if they represent features or models), and their
 * descriptions are generally deserialized using the corresponding classes
 * (for example ark.data.feature.Feature or ark.model.Model).  The deserialization
 * methods in these classes and in the experiment classes generally rely on 
 * the ark.util.SerializationUtil to parse each line.  This process is somewhat
 * messy and complicated and should probably be refactored sometime in the future.
 * 
 * The operations of a particular experiment are defined by the classes 
 * that extend Experiment (see other classes under ark.experiment)
 *
 * The output of the experiment is generally written to debug, 
 * results, model, and data
 * files in an output directory whose location is given by an 
 * ark.util.OutputWriter object stored in an ark.data.DataTools object
 * in an ark.data.annotation.Datum.Tools object.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 *
 */
public abstract class Experiment<D extends Datum<L>, L> {
	protected String name;
	protected String inputPath;
	protected Datum.Tools<D, L> datumTools;
	protected int maxThreads;
	
	public Experiment(String name, String inputPath, Datum.Tools<D, L> datumTools) {
		this.name = name;
		this.inputPath = inputPath;
		this.datumTools = datumTools;
	}
	
	protected abstract boolean execute();
	protected abstract boolean deserializeNext(BufferedReader reader, String nextName) throws IOException;
	
	public boolean run() {
		try {
			return deserialize() && execute();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	protected boolean deserialize() throws IOException {
		BufferedReader reader = FileUtil.getFileReader(this.inputPath);
		String assignmentLeft = null; // The name on the left hand side of the equals sign

		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("randomSeed"))
				this.datumTools.getDataTools().setRandomSeed(Long.valueOf(SerializationUtil.deserializeAssignmentRight(reader)));
			else if (assignmentLeft.equals("maxThreads"))
				this.maxThreads = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
			else if (!deserializeNext(reader, assignmentLeft))
				return false;
		}
		
		return true;
	}
}
