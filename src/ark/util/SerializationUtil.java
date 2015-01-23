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

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 *
 * SerializationUtil provides methods for serializing and deserializing 
 * model configurations and features, assuming that they are represented
 * by:
 * 
 * Lists of the form: l_1, l_2, l_3,..., l_n
 * Argument lists of the form: a_1=v_1, a_2=v_2, a_3=v_3,...,a_n=v_n
 * Assignments of the form: a=v
 * 
 * The methods for serializing and deserializing are written so that other
 * classes can call them on large inputs without running out of memory. That's
 * the main advantage of using this instead of something like JSON.  
 * 
 * See the ark.data.feature.Feature class for examples of how this is used.
 * 
 * @author Bill McDowell
 *
 */
public class SerializationUtil {
	public static Map<String, String> deserializeArguments(Reader reader) throws IOException {
		Map<String, String> arguments = new HashMap<String, String>();
		List<String> assignments = SerializationUtil.deserializeList(reader);
		for (String assignmentStr : assignments) {
			// Slow... optimize this if necessary
			Pair<String, String> assignment = SerializationUtil.deserializeAssignment(new StringReader(assignmentStr));
			if (assignment == null)
				return null;
			arguments.put(assignment.getFirst(), assignment.getSecond());
		}
				
		return arguments;
	}
	
	public static List<String> deserializeList(Reader reader) throws IOException {
		List<String> list = new ArrayList<String>();
		int cInt = reader.read();
		char c = (char)cInt;
		StringBuilder item = new StringBuilder();
		while (cInt != -1 && c != '\n' && c != '\t') {
			boolean inQuotes = false;
			while (cInt != -1 && (inQuotes || (c != ',' && c != '\n'&& c != ')'))) {
				item = item.append(c);
				cInt = reader.read();
				c = (char)cInt;
				if (c == '"')
					inQuotes = !inQuotes;
			}
			
			list.add(item.toString().trim());
			item = new StringBuilder();
			if (cInt != -1 && c != '\n') {
				cInt = reader.read();
				c = (char)cInt;
			}
		}
		
		return list;
	}
	
	public static Pair<String, String> deserializeAssignment(Reader reader) throws IOException {
		int cInt = reader.read();
		char c = (char)cInt;
		StringBuilder first = new StringBuilder();
		while (cInt != -1 && c != '=' && c != ')') {
			first = first.append(c);
			cInt = reader.read();
			c = (char)cInt;
		}
		
		if (cInt == -1 || first.length() == 0)
			return null;
		
		boolean inQuotes = false;
		cInt = reader.read();
		c = (char)cInt;
		if (c == '"')
			inQuotes = true;
		
		StringBuilder second = new StringBuilder();
		
		while (cInt != -1 && (inQuotes || (c != ',' && c != '\n' && c != ')'))) {
			second = second.append(c);
			cInt = reader.read();
			c = (char)cInt;
			if (c == '"')
				inQuotes = !inQuotes;
		}
		
		String secondStr = second.toString();
		if (secondStr.startsWith("\""))
			secondStr = secondStr.substring(1, secondStr.length()-1);
		
		return new Pair<String, String>(first.toString().trim(),secondStr.trim());
	}
	
	public static String deserializeGenericName(Reader reader) throws IOException {
		int cInt = -1;
		char c = 0;
		StringBuilder genericName = new StringBuilder();
		
		do {
			cInt = reader.read();
			c = (char)cInt;
			
			if (cInt != -1 && c != '(')
				genericName.append(c);
		} while (cInt != -1 && c != '(');
		
		return genericName.toString();
	}
	
	public static String deserializeAssignmentLeft(Reader reader) throws IOException {
		int cInt = -1;
		char c = 0;
		StringBuilder left = new StringBuilder();
		
		do {
			cInt = reader.read();
			c = (char)cInt;
			
			if (cInt != -1 && c != '=' && c != '\n' && c != '\t' && c != ' ')
				left.append(c);
		} while (cInt != -1 && c != '=' && (left.length() == 0 || (c != '\n' && c != '\t' && c != ' ')));
		
		return (cInt == -1 || c == '\n') ? null : left.toString().trim();
	}
	
	public static String deserializeAssignmentRight(Reader reader) throws IOException {
		int cInt = -1;
		char c = 0;
		StringBuilder right = new StringBuilder();
		
		do {
			cInt = reader.read();
			c = (char)cInt;
			
			if (cInt != -1 && c != '\n' && c != ')')
				right.append(c);
		} while (cInt != -1 && c != '\n' && c != ')');
		
		return right.toString().trim();
	}
	
	public static String deserializeString(Reader reader) throws IOException {
		int cInt = -1;
		char c = 0;
		StringBuilder str = new StringBuilder();
		boolean firstQuote = false;
		boolean secondQuote = false;
		do {
			cInt = reader.read();
			c = (char)cInt;
			
			if (c == '"') {
				if (!firstQuote) {
					firstQuote = true;
				} else if (!secondQuote) {
					secondQuote = true;
				}
			} else if (firstQuote) {
				str = str.append(c);
			}
		} while (!firstQuote || !secondQuote);
		
		return str.toString();
	}
	
	public static <T> boolean serializeArguments(Map<String, T> arguments, Writer writer) throws IOException {
		int i = 0;
		for (Entry<String, T> argument : arguments.entrySet()) {
			writer.write(argument.getKey() + "=" + argument.getValue());
			if (i != arguments.size() - 1)
				writer.write(",");
			i++;
		}
		
		return true;
	}
	
	public static <T> boolean serializeList(Iterable<T> iterable, Writer writer) throws IOException {		
		List<T> list = new ArrayList<T>();
		for (T item : iterable)
			list.add(item);
		
		for (int i = 0; i < list.size(); i++) {
			writer.write(list.get(i).toString());
			if (i != list.size() - 1)
				writer.write(",");
		}
		return true;
	}
	
	public static <T> boolean serializeAssignment(Pair<String, T> assignment, Writer writer) throws IOException {
		writer.write(assignment.getFirst() + "=" + assignment.getSecond());
		return true;
	}
}
