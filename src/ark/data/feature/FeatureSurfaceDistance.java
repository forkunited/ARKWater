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
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.nlp.TokenSpan;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

/**
 * FIXME Fill this in sometime maybe
 * 
 * @author Jesse Dodge
 *
 * @param <D>
 * @param <L>
 */
public class FeatureSurfaceDistance<D extends Datum<L>, L> extends Feature<D, L>{
	
	protected BidirectionalLookupTable<String, Integer> vocabulary;

	protected Datum.Tools.TokenSpanExtractor<D, L> sourceTokenExtractor;
	protected Datum.Tools.TokenSpanExtractor<D, L> targetTokenExtractor;
	protected String[] parameterNames = {"sourceTokenExtractor", "targetTokenExtractor", "PoS"};
	
	public FeatureSurfaceDistance(){
		vocabulary = new BidirectionalLookupTable<String, Integer>();
	}

	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		
		for (D datum : dataSet) {
			counter.incrementCount(findDistance(datum));
		}
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		return true;
	}
	
	private String findDistance(D datum){
		TokenSpan source = sourceTokenExtractor.extract(datum)[0];
		TokenSpan target = targetTokenExtractor.extract(datum)[0];
		return "" + (target.getStartTokenIndex() - source.getStartTokenIndex());
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<Integer, Double> vect = new HashMap<Integer, Double>();
		vect.put(vocabulary.get(findDistance(datum)), 1.0);
		return vect;
	}

	@Override
	public String getGenericName() {
		return "surfaceDistance";
	}

	@Override
	public int getVocabularySize() {
		return vocabulary.size();
	}

	@Override
	public String getVocabularyTerm(int index) {
		return "" + vocabulary.reverseGet(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		vocabulary.put(term, index);
		return true;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("sourceTokenExtractor")) 
			return (this.sourceTokenExtractor == null) ? null : this.sourceTokenExtractor.toString();
		else if (parameter.equals("targetTokenExtractor"))
			return (this.targetTokenExtractor == null) ? null : this.targetTokenExtractor.toString();
		return null;	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("sourceTokenExtractor")) 
			this.sourceTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("targetTokenExtractor")) 
			this.targetTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else
			return false;
		return true;

	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureSurfaceDistance<D,L>();
	}
	
	@Override
	protected <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects) {
		if (!newObjects) {
			FeatureSurfaceDistance<D1,L1> cloneFeature = (FeatureSurfaceDistance<D1, L1>)clone;
			cloneFeature.vocabulary = this.vocabulary;
		}
		
		return true;
	}
	
	@Override
	protected boolean serializeHelper(Writer writer) {
		return true;
	}
	
	@Override
	protected boolean deserializeHelper(BufferedReader writer) {
		return true;
	}
}
