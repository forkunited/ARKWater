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
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

/**
 * 
 * @author jesse
 *
 * @param <D> datum. probably like a tlinkable.
 * @param <L>
 * @param PoS: indicates the parts of speech of the words we should extract. 
 * @param startWindowRelativeIndex: the start of the window from which we will extract words of type PoS. relative to the start of the tokenSpan.
 * 									can be set to -1 to indicate 'from the start of the sentence.'
 * @param endWindowRelativeIndex: the end of the window from whicch we will extract words of type PoS. relative to the start of the tokenSpan.
 * 								  can be set to -1 to indicate 'till the end of the sentence.'
 */

public class FeatureNGramPoS<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected String PoS;
	protected int tokensBeforeTokenSpan;
	protected int tokensAfterTokenSpan;
	protected String[] parameterNames = {"minFeatureOccurrence", "tokenExtractor", "PoS", "tokensBeforeTokenSpan", "tokensAfterTokenSpan"};
	
	public FeatureNGramPoS(){
		vocabulary = new BidirectionalLookupTable<String, Integer>();
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Set<String> nGramPoS = getNGramPoSForDatum(datum);
			for (String ngram : nGramPoS) {
				counter.incrementCount(ngram);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	private Set<String> getNGramPoSForDatum(D datum){
		Set<String> nGramPoS = new HashSet<String>();
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		
		for (TokenSpan tokenSpan : tokenSpans) {
			if (tokenSpan.getStartTokenIndex() < 0){
				return nGramPoS;
			}
			int sentLength = tokenSpan.getDocument().getSentenceTokenCount(tokenSpan.getSentenceIndex());
			
			int sentIndex = tokenSpan.getSentenceIndex();
			
			for (int tokenIndex = 0 ; tokenIndex < sentLength; tokenIndex++){
				// if the tokenIndex is before the tokenSpan, and (the tokensBeforeTokenSpan == -1 or
				// 												   tokenSpan.getStartTokenIndex - tokenIndex <= tokensBeforeTokenSpan)
				if (tokenIndex < tokenSpan.getStartTokenIndex() && (tokensBeforeTokenSpan == -1 || 
						tokenSpan.getStartTokenIndex() - tokenIndex <= tokensBeforeTokenSpan)){
					if (tokenSpan.getDocument().getPoSTag(sentIndex, tokenIndex).toString().equals(PoS)){
						nGramPoS.add(tokenSpan.getDocument().getToken(sentIndex, tokenIndex));
					}
				} else if(tokenIndex > tokenSpan.getEndTokenIndex() && (tokensAfterTokenSpan == -1 || 
						tokenIndex - tokenSpan.getEndTokenIndex() <= tokensAfterTokenSpan)){
					if (tokenSpan.getDocument().getPoSTag(sentIndex, tokenIndex).toString().equals(PoS)){
						nGramPoS.add(tokenSpan.getDocument().getToken(sentIndex, tokenIndex));
					}
				}

			}
		}
		
		return nGramPoS;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> posForDatum = getNGramPoSForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String ngramPoS : posForDatum) {
			if (this.vocabulary.containsKey(ngramPoS))
				vector.put(this.vocabulary.get(ngramPoS), 1.0);		
		}

		return vector;
	}


	@Override
	public String getGenericName() {
		return "NGramPoS";
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
	public String getParameterValue(String parameter) {
		if (parameter.equals("minFeatureOccurrence")) 
			return String.valueOf(this.minFeatureOccurrence);
		else if (parameter.equals("tokenExtractor"))
			return (this.tokenExtractor == null) ? null : this.tokenExtractor.toString();
		else if (parameter.equals("PoS"))
			return this.PoS;
		return null;
	}
	
	// note these will be called by TLinkDatum.Tools, and in that class TargetTokenSpan exists, for example.
	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("PoS"))
			this.PoS = parameterValue;
		else
			return false;
		return true;
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureNGramPoS<D, L>();
	}

	@Override
	protected <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects) {
		if (!newObjects) {
			FeatureNGramPoS<D1,L1> cloneFeature = (FeatureNGramPoS<D1, L1>)clone;
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
