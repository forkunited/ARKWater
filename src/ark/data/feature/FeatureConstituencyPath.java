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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.ConstituencyParse.ConstituentPath;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;
import ark.util.ThreadMapper;

/**
 * FeatureConstituencyPath computes paths in constituency parse trees
 * between token spans
 * associated with a datum. For a datum d with source token-span extractor S,
 * and target token span extractor T, the feature computes vector:
 * 
 * <1(p_1 \in P(S(d),T(d)), 1(p_2 \in P(S(d),T(d))), ... , 1(p_n \in P(S(d),T(d)))>
 * 
 * Where P(S(d),T(d)) gives the set of shortest constituency paths between token spans
 * in S(d) and token spans in T(d), and p_i is a constituency path in the vocabulary
 * of possible paths from the full data set containing d.
 *  
 * The 'minFeatureOccurrence' parameter determines the minimum number of times a
 * path p_i must appear in the full data set for it to have a component in the 
 * returned vectors.
 * 
 * The 'useRelationTypes' parameter determines whether the constituency paths corresponding
 * to components in the returned vector should be typed.
 * 
 * @author Jesse Dodge, Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureConstituencyPath<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected Datum.Tools.TokenSpanExtractor<D, L> sourceTokenExtractor;
	protected Datum.Tools.TokenSpanExtractor<D, L> targetTokenExtractor;
	protected boolean useRelationTypes = true;
	protected String[] parameterNames = {"minFeatureOccurrence", "sourceTokenExtractor", "targetTokenExtractor", "useRelationTypes"};
	
	public FeatureConstituencyPath() {
		
	}
	
	public FeatureConstituencyPath(Context<D, L> context) {
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
		this.context = context;
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		final CounterTable<String> counter = new CounterTable<String>();
		dataSet.map(new ThreadMapper.Fn<D, Boolean>() {
			@Override
			public Boolean apply(D datum) {
				Set<String> paths = getPathsForDatum(datum);
				for (String path : paths) {
					counter.incrementCount(path);
				}
				return true;
			}
		});
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	private Set<String> getPathsForDatum(D datum){
		Set<String> paths = new HashSet<String>();
		
		TokenSpan[] sourceTokenSpans = this.sourceTokenExtractor.extract(datum);
		TokenSpan[] targetTokenSpans = this.targetTokenExtractor.extract(datum);
		
		for (TokenSpan sourceSpan : sourceTokenSpans) {
			for (TokenSpan targetSpan : targetTokenSpans){
				ConstituentPath path = getShortestPath(sourceSpan, targetSpan);
				if (path == null)
					continue;
				paths.add(path.toString(this.useRelationTypes));
			}
		}
		return paths;
	}
	
	private ConstituentPath getShortestPath(TokenSpan sourceSpan, TokenSpan targetSpan){
		if (sourceSpan.getSentenceIndex() < 0 
				|| targetSpan.getSentenceIndex() < 0 
				|| sourceSpan.getSentenceIndex() != targetSpan.getSentenceIndex())
			return null;
		
		ConstituentPath shortestPath = null;
		int sentenceIndex = sourceSpan.getSentenceIndex();
		ConstituencyParse parse = sourceSpan.getDocument().getConstituencyParse(sentenceIndex);
		for (int i = sourceSpan.getStartTokenIndex(); i < sourceSpan.getEndTokenIndex(); i++){
			for (int j = targetSpan.getStartTokenIndex(); j < targetSpan.getEndTokenIndex(); j++){
				ConstituentPath path = parse.getPath(i, j);
				if (shortestPath == null || (path != null && path.getLength() < shortestPath.getLength()))
					shortestPath = path;
			}
		}

		return shortestPath;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> pathsForDatum = getPathsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String path : pathsForDatum) {
			if (this.vocabulary.containsKey(path))
				vector.put(this.vocabulary.get(path), 1.0);		
		}

		return vector;
	}


	@Override
	public String getGenericName() {
		return "ConstituencyPath";
	}
	
	@Override
	public String getVocabularyTerm(int index) {
		return this.vocabulary.reverseGet(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		this.vocabulary.put(term, index);
		return true;
	}

	@Override
	public int getVocabularySize() {
		return this.vocabulary.size();
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("minFeatureOccurrence")) 
			return Obj.stringValue(String.valueOf(this.minFeatureOccurrence));
		else if (parameter.equals("sourceTokenExtractor"))
			return Obj.stringValue((this.sourceTokenExtractor == null) ? "" : this.sourceTokenExtractor.toString());
		else if (parameter.equals("targetTokenExtractor"))
			return Obj.stringValue((this.targetTokenExtractor == null) ? "" : this.targetTokenExtractor.toString());
		else if (parameter.equals("useRelationTypes"))
			return Obj.stringValue(String.valueOf(this.useRelationTypes));
		return null;
	}
	
	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("sourceTokenExtractor"))
			this.sourceTokenExtractor = this.context.getDatumTools().getTokenSpanExtractor(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("targetTokenExtractor"))
			this.targetTokenExtractor = this.context.getDatumTools().getTokenSpanExtractor(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("useRelationTypes"))
			this.useRelationTypes = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	public Feature<D, L> makeInstance(Context<D, L> context) {
		return new FeatureConstituencyPath<D, L>(context);
	}

	@Override
	protected <T extends Datum<Boolean>> Feature<T, Boolean> makeBinaryHelper(
			Context<T, Boolean> context, LabelIndicator<L> labelIndicator,
			Feature<T, Boolean> binaryFeature) {
		FeatureConstituencyPath<T, Boolean> binaryFeatureConst = (FeatureConstituencyPath<T,Boolean>)binaryFeature;
		
		binaryFeatureConst.vocabulary = this.vocabulary;
		
		return binaryFeatureConst;
	}

	@Override
	protected boolean fromParseInternalHelper(AssignmentList internalAssignments) {
		return true;
	}

	@Override
	protected AssignmentList toParseInternalHelper(
			AssignmentList internalAssignments) {
		return internalAssignments;
	}
	
}
