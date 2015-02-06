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
import java.util.Arrays;
import java.util.List;

import ark.cluster.Clusterer;
import ark.data.annotation.Datum;
import ark.data.annotation.Document;
import ark.data.annotation.nlp.TokenSpan;

/**
 * 
 * FeatureNGram computes n-gram features for datums.  For
 * a datum d and token-extractor T, and scaling function s:R->R the feature computes 
 * the vector:
 * 
 * <s(1(v_1\in F(T(d))))), s(1(v_2 \in F(T(d)))), ... , s(1(v_n \in F(T(d))))>
 * 
 * Where F(T(d)) is a subset of the tokens given by T(d) that 
 * depends on the 
 * particular n-gram feature that is being computed (e.g. NGramContext,
 * NGramDep, etc), and v_i 
 * is an n-gram in vocabulary of possible n-grams from the full
 * data-set.  
 * 
 * For examples of possible F, see the feature types that extend 
 * this class.  Possibilities for s are given by the Scale enum 
 * defined below.
 * 
 * The minFeatureOccurrence parameter determineds the minimum number
 * of times an n-gram must occur in the data-set for it to be included
 * as a component in the computed vectors.
 * 
 * The cleanFn parameter is a string cleaning function that is applied to
 * each gram in each n-gram before the vectors are computed.
 * 
 * Optionally, if a clusterer parameter is provided, then grams of
 * the n-grams are first mapped to their clusters
 * 
 * @author Bill McDowell
 * 
 * @param <D> datum type
 * @param <L> datum label type
 *
 */
public abstract class FeatureNGram<D extends Datum<L>, L> extends FeatureGram<D, L> {	
	protected int n;
	protected Clusterer<TokenSpan> clusterer;
	
	public FeatureNGram() {
		super();
		
		this.n = 1;
		this.clusterer = null;
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 2);
		this.parameterNames[this.parameterNames.length - 2] = "clusterer";
		this.parameterNames[this.parameterNames.length - 1] = "n";
	}

	protected List<String> getCleanNGramsAtPosition(Document document, int sentenceIndex, int startTokenIndex) {
		if (this.clusterer != null) {
			TokenSpan ngramSpan = new TokenSpan(document, sentenceIndex, startTokenIndex, startTokenIndex + this.n);
			List<String> clusters = this.clusterer.getClusters(ngramSpan);
			if (clusters == null)
				return new ArrayList<String>();
			else
				return clusters;
		} 
		
		StringBuilder ngram = new StringBuilder();		
		for (int i = startTokenIndex; i < startTokenIndex + this.n; i++) {
			String cleanGram = this.cleanFn.transform(document.getToken(sentenceIndex, i));
			if (cleanGram.length() == 0)
				continue;
			ngram.append(cleanGram).append("_");
		}
		
		List<String> ngrams = new ArrayList<String>();
		if (ngram.length() == 0)
			return ngrams;
		
		ngram = ngram.delete(ngram.length() - 1, ngram.length());	
		ngrams.add(ngram.toString());
		
		return ngrams;
	}

	@Override
	public String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("clusterer"))
			return (this.clusterer == null) ? "None" : this.clusterer.getName();
		else if (parameter.equals("n"))
			return String.valueOf(this.n);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("clusterer"))
			this.clusterer = datumTools.getDataTools().getTokenSpanClusterer(parameterValue);
		else if (parameter.equals("n"))
			this.n = Integer.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}
}
