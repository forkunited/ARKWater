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

package ark.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * 
 * OutputWriter provides methods for writing output to several file
 * streams.  It's useful when training and evaluating models, and you want 
 * to write logging information, results, data, and models to separate files
 * while running experiments.
 * 
 * @author Bill McDowell
 *
 */
public class OutputWriter {
	private File debugFile;
	private File resultsFile;
	private File dataFile;
	private File modelFile;
	
	private BufferedWriter debugWriter;
	private BufferedWriter resultsWriter;
	private BufferedWriter dataWriter;
	private BufferedWriter modelWriter;
	
	public OutputWriter() {
		this.debugWriter = null;
		this.resultsWriter = null;
		this.dataWriter = null;
		this.modelWriter = null;
	}
	
	public OutputWriter(File debugFile, File resultsFile, File dataFile, File modelFile) {
		try {
			this.debugFile = debugFile;
			this.resultsFile = resultsFile;
			this.dataFile = dataFile;
			this.modelFile = modelFile;
			
			this.debugWriter = new BufferedWriter(new FileWriter(debugFile.getAbsolutePath()));
			this.resultsWriter = new BufferedWriter(new FileWriter(resultsFile.getAbsolutePath()));
			this.dataWriter = new BufferedWriter(new FileWriter(dataFile.getAbsolutePath()));
			this.modelWriter = new BufferedWriter(new FileWriter(modelFile.getAbsolutePath()));
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public String getDebugFilePath() {
		return this.debugFile.getAbsolutePath();
	}
	
	public String getResultsFilePath() {
		return this.resultsFile.getAbsolutePath();
	}
	
	public String getDataFilePath() {
		return this.dataFile.getAbsolutePath();
	}
	
	public String getModelFilePath() {
		return this.modelFile.getAbsolutePath();
	}
	
	public void debugWriteln(String str) {
		debugWrite(str + "\n");
	}
	
	public void debugWrite(String str) {
		System.out.print(str);
		System.out.flush();
		if (this.debugWriter == null)
			return;
			
		try {
			synchronized (this.debugWriter) {
				this.debugWriter.write(str);
				this.debugWriter.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public void resultsWriteln(String str) {
		resultsWrite(str + "\n");
	}
	
	public void resultsWrite(String str) {
		if (this.resultsWriter == null) {
			System.out.print(str);
			System.out.flush();
			return;
		}
		
		try {
			synchronized (this.resultsWriter) {
				this.resultsWriter.write(str);
				this.resultsWriter.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public void dataWriteln(String str) {
		dataWrite(str + "\n");
	}
	
	public void dataWrite(String str) {
		if (this.dataWriter == null) {
			System.out.print(str);
			System.out.flush();
			return;
		}
		
		try {
			synchronized (this.dataWriter) {
				this.dataWriter.write(str);
				this.dataWriter.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public void modelWriteln(String str) {
		modelWrite(str + "\n");
	}
	
	public void modelWrite(String str) {
		if (this.modelWriter == null) {
			System.out.print(str);
			System.out.flush();
			return;
		}
		
		try {
			synchronized (this.modelWriter) {
				this.modelWriter.write(str);
				this.modelWriter.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public void close() {
		try {
			if (this.debugWriter != null)
				this.debugWriter.close();
			if (this.dataWriter != null)
				this.dataWriter.close();
			if (this.resultsWriter != null)
				this.resultsWriter.close();
			if (this.modelWriter != null)
				this.modelWriter.close();
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
}
