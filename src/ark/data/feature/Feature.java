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

package ark.data.feature;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.util.Pair;
import ark.util.Parameterizable;
import ark.util.SerializationUtil;

/**
 * Feature represents an abstract feature to be computed on data and
 * used in a model.
 * 
 * Implementations of particular features derive from the Feature class,
 * and the Feature class is primarily responsible for providing the
 * methods necessary for deserializing features from
 * configuration files.  The features are defined by strings in the 
 * configuration file of the form:
 * 
 * feature(_[featureReferenceName](_ignored))=[featureGenericName]([parameterName1]=[parameterValue1],...)
 * 
 * Where strings in parentheses are optional and strings in square brackets
 * are replaced by feature specific information.
 * 
 * A feature's computed values are generally vectors of real values.  
 * Each component of a feature's vector has a name, and the set of all names 
 * for components is the feature's 'vocabulary'.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class Feature<D extends Datum<L>, L> implements Parameterizable<D, L> {
	// particular name by which the feature is referenced in configuration files
	private String referenceName; 
	// indicator of whether to ignore the feature in data sets so that it
	// isn't included in models
	private boolean ignored;
	
	/**
	 * @param dataSet
	 * @return true if the feature has been initialized for the dataSet
	 */
	public abstract boolean init(FeaturizedDataSet<D, L> dataSet);
	
	/**
	 * @param datum
	 * @return a sparse mapping from vector indices to values of the feature
	 * for the given datum.
	 */
	public abstract Map<Integer, Double> computeVector(D datum);
	
	/**
	 * @return the generic name of the feature in the configuration files.  For
	 * feature class Feature[X], the generic name should usually be X.
	 */
	public abstract String getGenericName();
	
	/**
	 * @return the length of the vector computed by this feature for each
	 * datum
	 */
	public abstract int getVocabularySize();
	
	/**
	 * 
	 * @param index
	 * @return the name of the component at the given index within vectors
	 * computed by this feature
	 * 
	 */
	public abstract String getVocabularyTerm(int index); 
	
	/**
	 * 
	 * @param index
	 * @param term
	 * @return true if the name of component at the given index has been set
	 * to the value of term.  This is used when deserializing features that 
	 * were previously computed and saved.
	 * 
	 */
	protected abstract boolean setVocabularyTerm(int index, String term);

	/**
	 * @return a generic instance of the feature.  This is used when deserializing
	 * the parameters for the feature from a configuration file
	 */
	public abstract Feature<D, L> makeInstance();
	
	
	/**
	 * @param clone
	 * @param newObjects indicates whether new objects should be constructed or use
	 * same objects in clone.  If not new objects, then cloneHelper is responsible
	 * for copying the vocabulary object references into the clone 
	 * @return true if clone has been given remaining non-parameter
	 * /non-vocabulary properties of feature 
	 */
	protected abstract <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects);
	
	/**
	 * @param serializeHelper
	 * @return true if remaining non-parameter/non-vocabulary
	 * properties of feature have been serialized
	 */
	protected abstract boolean serializeHelper(Writer writer) throws IOException;
	
	/**
	 * @param deserializeHelper
	 * @return true if remaining non-parameter/non-vocabulary
	 * properties of feature have been deserialized
	 */
	protected abstract boolean deserializeHelper(BufferedReader writer) throws IOException;
	
	/**
	 * @return a name by which this particular feature is referenced by other
	 * features in experiment configuration files.  This feature can be retrieved
	 * from a FeaturizedDataSet using this name.
	 * 
	 */
	public String getReferenceName() {
		return this.referenceName;
	}
	
	/**
	 * @return true if this feature should be ignored by models (it is only used for the
	 * computation of other features)
	 */
	public boolean isIgnored() {
		return this.ignored;
	}
	
	public Map<Integer, String> getSpecificShortNamesForIndices(Iterable<Integer> indices) {
		String prefix = getSpecificShortNamePrefix();
		Map<Integer, String> specificShortNames = new HashMap<Integer, String>();
		for (Integer index : indices) {
			specificShortNames.put(index, prefix + getVocabularyTerm(index));
		}
		
		return specificShortNames;
	}
	
	public Map<Integer, String> getVocabularyForIndices(Iterable<Integer> indices) {
		Map<Integer, String> vocabulary = new HashMap<Integer, String>();
		for (Integer index : indices) {
			vocabulary.put(index, getVocabularyTerm(index));
		}
		
		return vocabulary;
	}
	
	public List<String> getSpecificShortNames() {
		String prefix = getSpecificShortNamePrefix();
		int vocabularySize = getVocabularySize();
		List<String> specificShortNames = new ArrayList<String>(vocabularySize);
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			specificShortNames.add(prefix + ((vocabularyTerm == null) ? "" : vocabularyTerm));
		}
		
		return specificShortNames;
	}
	
	public <D1 extends Datum<L1>, L1> Feature<D1, L1> clone(Datum.Tools<D1, L1> datumTools) {
		return clone(datumTools, null, true);
	}
	
	public <D1 extends Datum<L1>, L1> Feature<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment) {
		return clone(datumTools, environment, true);
	}
	
	public <D1 extends Datum<L1>, L1> Feature<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment, boolean newObjects) {
		Feature<D1, L1> clone = datumTools.makeFeatureInstance(getGenericName(), true);
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (environment != null && parameterValue != null) {
				for (Entry<String, String> entry : environment.entrySet())
					parameterValue = parameterValue.replace("--" + entry.getKey() + "--", entry.getValue());
			}
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		
		clone.referenceName = this.referenceName;
		clone.ignored = this.ignored;
		
		if (newObjects) { 
			int vocabularySize = getVocabularySize();
			for (int i = 0; i < vocabularySize; i++) {
				clone.setVocabularyTerm(i, getVocabularyTerm(i));
			}
		}
		
		if (!cloneHelper(clone, newObjects))
			return null;
		
		return clone;
	}

	public boolean deserialize(BufferedReader reader, boolean readGenericName, boolean readVocabulary, Datum.Tools<D, L> datumTools, String referenceName, boolean ignored) throws IOException {		
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet())
				this.setParameterValue(entry.getKey(), entry.getValue(), datumTools);
		
		if (readVocabulary) {
			Map<String, String> vocabulary = SerializationUtil.deserializeArguments(reader);
			if (vocabulary == null) {
				datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Feature failed to deserialize vocabulary.");
				return false;
			}
			
			for (Entry<String, String> entry : vocabulary.entrySet()) {
				if (!setVocabularyTerm(Integer.valueOf(entry.getValue()), entry.getKey())) {
					datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Feature failed to set vocabulary terms.");
					return false;
				}
			}
			
			if (!deserializeHelper(reader)) {
				datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Feature failed to deserialize non-vocabulary initialization data.");
				return false;
			}
		}

		this.referenceName = referenceName;
		this.ignored = ignored;
		
		return true;
	}
	
	public boolean serialize(Writer writer, boolean includeReferenceName) throws IOException {
		if (includeReferenceName) {
			writer.write("feature" + 
					(this.referenceName == null ? "" : "_" + this.referenceName) + 
					((!this.ignored) ? "" : "_ignored")
					+ "=");
		}
		
		int vocabularySize = getVocabularySize();
		writer.write(toString(false));
		writer.write("\t");
		
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			if (vocabularyTerm == null)
				continue;
			Pair<String, Integer> v = new Pair<String, Integer>(vocabularyTerm, i);
			if (!SerializationUtil.serializeAssignment(v, writer))
				return false;
			if (i != vocabularySize - 1)
				writer.write(",");
		}
		
		writer.write(")\t");
		
		if (!serializeHelper(writer))
			return false;
		
		return true;
	}
	
	public String toString(boolean withVocabulary) {
		if (withVocabulary) {
			StringWriter stringWriter = new StringWriter();
			try {
				if (serialize(stringWriter, withVocabulary))
					return stringWriter.toString();
				else
					return null;
			} catch (IOException e) {
				return null;
			}
		} else {
			String genericName = getGenericName();
			Map<String, String> parameters = new HashMap<String, String>();
			String[] parameterNames = getParameterNames();
			for (int i = 0; i < parameterNames.length; i++)
				parameters.put(parameterNames[i], "\"" + getParameterValue(parameterNames[i]) + "\"");
			StringWriter parametersWriter = new StringWriter();
			
			try {
				SerializationUtil.serializeArguments(parameters, parametersWriter);
			} catch (IOException e) {
				return null;
			}
			
			String parametersStr = parametersWriter.toString();
			return genericName + "(" + parametersStr + ")";
		}
	}
	
	public String toString() {
		return toString(false);
	}
	
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, false, datumTools, null, false);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	protected String getSpecificShortNamePrefix() {
		if (this.referenceName != null)
			return this.referenceName + "_";
		
		StringBuilder shortNamePrefixBuilder = new StringBuilder();
		String genericName = shortenName(getGenericName());
		String[] parameterNames = getParameterNames();
		
		shortNamePrefixBuilder = shortNamePrefixBuilder.append(genericName).append("_");
		for (int i = 0; i < parameterNames.length; i++)
			shortNamePrefixBuilder = shortNamePrefixBuilder.append(shortenName(parameterNames[i]))
														.append("-")
														.append(getParameterValue(parameterNames[i]))
														.append("_");
		
		return shortNamePrefixBuilder.toString();
	}
	
	private String shortenName(String name) {
		if (name.length() == 0)
			return name;
		
		StringBuilder shortenedName = new StringBuilder();
		shortenedName.append(name.charAt(0));
		
		int curWordSize = 0;
		for (int i = 1; i < name.length(); i++) {
			if (Character.isUpperCase(name.charAt(i))) {
				shortenedName.append(name.charAt(i));
				curWordSize = 1;
			} else if (curWordSize < 4) {
				shortenedName.append(name.charAt(i));
				curWordSize++;
			}
		}
		
		return shortenedName.toString();
	}
	
	public static <D extends Datum<L>, L> Feature<D, L> deserialize(BufferedReader reader, boolean readParameters, Datum.Tools<D, L> datumTools) throws IOException {		
		String assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader);
		if (assignmentLeft == null)
			return null;
	
		String[] nameParts = assignmentLeft.split("_");
		String referenceName = null;
		boolean ignore = false;
		if (nameParts.length > 1)
			referenceName = nameParts[1];
		if (nameParts.length > 2)
			ignore = true;
		
		String featureName = SerializationUtil.deserializeGenericName(reader);
		Feature<D, L> feature = datumTools.makeFeatureInstance(featureName);
		if (!feature.deserialize(reader, false, readParameters, datumTools, referenceName, ignore))
			return null;

		return feature;
	}
}
