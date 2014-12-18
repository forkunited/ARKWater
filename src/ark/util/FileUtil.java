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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;

/**
 * 
 * FileUtil helps deal with files on a typical file system or 
 * HDFS (in Hadoop).  Given a path for which to read/write, it
 * determines whether the path points to an HDFS or local
 * file system, and then performs operations accordingly.
 * 
 * @author Bill McDowell
 *
 */
public class FileUtil {
	public static BufferedReader getFileReader(String path) {
		File localFile = new File(path);
		try {
			if (localFile.exists())
				return new BufferedReader(new FileReader(localFile));
			else 
				System.err.println("WARNING: FileUtil failed to read file at " + path); // Do something better later
		} catch (Exception e) { System.err.println("WARNING: FileUtil failed to read file at " + path); e.printStackTrace(); }
		return HadoopUtil.getFileReader(path);
	}
	
	public static BufferedReader getFileReader(String path, String encoding) {
		File localFile = new File(path);
		try {
			if (localFile.exists())
				return new BufferedReader(new InputStreamReader(new FileInputStream(path), encoding));
			else 
				System.err.println("WARNING: FileUtil failed to read file at " + path); // Do something better later
		} catch (Exception e) { }
		
		return null;
	}
	
	public static BufferedReader getPropertiesReader(String[] possiblePaths) {
		for (int i = 0; i < possiblePaths.length; i++) {
			BufferedReader r = getFileReader(possiblePaths[i]);
			if (r != null)
				return r;
		}
		return null;
	}
}
