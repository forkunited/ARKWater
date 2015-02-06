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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Document;
import ark.data.annotation.nlp.TokenSpan;

/**
 * For each datum d FeatureNGramSentence computes a
 * vector:
 * 
 * <c(v_1\in S(T(d))), c(v_2 \in S(T(d))), ... , c(v_n \in S(T(d)))>
 * 
 * Where T is a token extractor, S(T(d)) computes the n-grams 
 * in the sentence surrounding the tokens given by T(d) in a source text document,
 * and c(v \in S(T(d))) computes the number of occurrences of n-gram v in S.  
 * The resulting
 * vector is given to methods in ark.data.feature.FeatureNGram to be normalized
 * and scaled in some way.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureNGramSentence<D extends Datum<L>, L> extends FeatureNGram<D, L> {
	private boolean noTokenSpan;
	
	public FeatureNGramSentence() {
		super();
		
		this.noTokenSpan = false;
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "noTokenSpan";
	}
	
	@Override
	protected Map<String, Integer> getGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Map<String, Integer> retNgrams = new HashMap<String, Integer>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			if (tokenSpan.getSentenceIndex() < 0)
				continue;
			
			Document document = tokenSpan.getDocument();
			int sentenceIndex = tokenSpan.getSentenceIndex();
			for (int i = 0; i < document.getSentenceTokenCount(sentenceIndex)-this.n+1; i++) {
				if (this.noTokenSpan && (
						(tokenSpan.containsToken(sentenceIndex, i))
						|| (tokenSpan.containsToken(sentenceIndex, i + this.n - 1))
						|| (i < tokenSpan.getStartTokenIndex() && i + this.n - 1 >= tokenSpan.getEndTokenIndex())
						))
					continue;
				
				List<String> ngrams = getCleanNGramsAtPosition(document, sentenceIndex, i);
				if (ngrams != null) {
					for (String ngram : ngrams) {
						if (this.n == 1) {
							// FIXME: This further splits unigrams if 
							// the cleaning process inserts spaces into
							// them.  It's currently a hack for the
							// text classification project, but it should
							// be done differently. It shouldn't affect
							// anything if the clean function never inserts
							// spaces
							String[] ngramParts = ngram.split("\\s+");
							for (String ngramPart : ngramParts) {
								if (!retNgrams.containsKey(ngramPart))
									retNgrams.put(ngramPart, 1);
								else
									retNgrams.put(ngramPart, retNgrams.get(ngramPart) + 1);
							}
						} else {
							if (!retNgrams.containsKey(ngram))
								retNgrams.put(ngram, 1);
							else
								retNgrams.put(ngram, retNgrams.get(ngram) + 1);
						}
					}
				}
			}
		}
		return retNgrams;
	}

	@Override
	public String getGenericName() {
		return "NGramSentence";
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureNGramSentence<D, L>();
	} 
	
	@Override
	public String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("noTokenSpan"))
			return String.valueOf(this.noTokenSpan);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("noTokenSpan"))
			this.noTokenSpan = Boolean.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}

}
