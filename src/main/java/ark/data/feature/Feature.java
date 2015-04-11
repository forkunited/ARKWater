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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.parse.ARKParsableFunction;
import ark.parse.Assignment;
import ark.parse.AssignmentList;
import ark.parse.Obj;

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
public abstract class Feature<D extends Datum<L>, L> extends ARKParsableFunction {
	protected Context<D, L> context;
	
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
	
	protected abstract <T extends Datum<Boolean>> Feature<T, Boolean> makeBinaryHelper(Context<T, Boolean> context, LabelIndicator<L> labelIndicator, Feature<T, Boolean> binaryFeature);
	
	protected abstract boolean fromParseInternalHelper(AssignmentList internalAssignments);
	
	protected abstract AssignmentList toParseInternalHelper(AssignmentList internalAssignments);
	
	/**
	 * @return a generic instance of the feature.  This is used when deserializing
	 * the parameters for the feature from a configuration file
	 */
	public abstract Feature<D, L> makeInstance(Context<D, L> context);
	
	/**
	 * @return true if this feature should be ignored by models (it is only used for the
	 * computation of other features)
	 */
	public boolean isIgnored() {
		return this.getModifiers().contains("ignored");
	}
	
	public Map<Integer, String> getSpecificShortNamesForIndices(Iterable<Integer> indices) {
		String prefix = getReferenceName();
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
		String prefix = getReferenceName();
		int vocabularySize = getVocabularySize();
		List<String> specificShortNames = new ArrayList<String>(vocabularySize);
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			specificShortNames.add(prefix + ((vocabularyTerm == null) ? "" : vocabularyTerm));
		}
		
		return specificShortNames;
	}
	
	public Feature<D, L> clone() {
		Feature<D, L> clone = this.context.getDatumTools().makeFeatureInstance(getGenericName(), this.context);
		if (!clone.fromParse(getModifiers(), getReferenceName(), toParse()))
			return null;
		return clone;
	}
	
	public <T extends Datum<Boolean>> Feature<T, Boolean> makeBinary(Context<T, Boolean> context, LabelIndicator<L> labelIndicator) {
		Feature<T, Boolean> binaryFeature = context.getDatumTools().makeFeatureInstance(getGenericName(), context);
		
		binaryFeature.referenceName = this.referenceName;
		binaryFeature.modifiers = this.modifiers;
		
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++)
			binaryFeature.setParameterValue(parameterNames[i], getParameterValue(parameterNames[i]));
		
		return makeBinaryHelper(context, labelIndicator, binaryFeature);
	}
	
	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {
		if (internalAssignments == null)
			return true;
		if (!internalAssignments.contains("vocabulary"))
			return false;
		
		Obj.Array vocabulary = (Obj.Array)internalAssignments.get("vocabulary").getValue();
		for (int i = 0; i < vocabulary.size(); i++) {
			if (!setVocabularyTerm(i, vocabulary.getStr(i)))
				return false;
		}
		
		return fromParseInternalHelper(internalAssignments);
	}
	
	@Override
	protected AssignmentList toParseInternal() {
		AssignmentList internalAssignments = new AssignmentList();
		
		Obj.Array vocabulary = new Obj.Array();
		for (int i = 0; i < getVocabularySize(); i++) {
			vocabulary.add(Obj.stringValue(getVocabularyTerm(i)));
		}
		
		if (vocabulary.size() > 0) {
			internalAssignments.add(
					Assignment.assignmentTyped(new ArrayList<String>(), Context.ARRAY_STR, "vocabulary", vocabulary)
			);
		}
		
		internalAssignments = toParseInternalHelper(internalAssignments);
		
		return (internalAssignments.size() == 0) ? null : internalAssignments;
	}
}