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
import ark.data.annotation.nlp.DependencyParse;

/**
 * For each datum d FeatureNGramDep computes a
 * vector:
 * 
 * <c(v_1\in D_p(T(d))), c(v_2 \in D_p(T(d))), ... , c(v_n \in D_p(T(d)))>
 * 
 * Where T is a token extractor, D_p(T(d)) computes n-grams related to tokens
 * given by T(d) in the dependency parse trees containing elements of T(d) in
 * a source document.  The particular way in which the n-grams in D_p(T(d))
 * are related to tokens in T(d) in a dependency parse tree depends on parameters
 * p.  Currently, p can specify that the n-grams in D_p(T(d)) must be immediate
 * children in the trees of T(d) or immediate parents in the trees of T(d).
 * The function c(v \in S) computes the number of occurrences of n-gram v in S.  
 * The resulting vector is given to methods in ark.data.feature.FeatureNGram to 
 * be normalized and scaled in some way.
 * 
 * The 'mode' parameter useRelationTypes determines whether different typed relations
 * in the dependency tree should have their own corresponding sets of components
 * in the returned vector.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureNGramDep<D extends Datum<L>, L> extends FeatureNGram<D, L> {
	public enum Mode {
		ParentsOnly,
		ChildrenOnly,
		ParentsAndChildren
	}
	
	private FeatureNGramDep.Mode mode;
	private boolean useRelationTypes; // include dependency relation types
	
	public FeatureNGramDep() {
		super();
		
		this.mode = Mode.ParentsAndChildren;
		this.useRelationTypes = true;
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 2);
		this.parameterNames[this.parameterNames.length - 2] = "mode";
		this.parameterNames[this.parameterNames.length - 1] = "useRelationTypes";
	}
	
	@Override
	protected Map<String, Integer> getGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Map<String, Integer> retNgrams = new HashMap<String, Integer>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			if (tokenSpan.getSentenceIndex() < 0)
				continue;
			
			Document document = tokenSpan.getDocument();
			int startIndex = tokenSpan.getStartTokenIndex();
			int endIndex = tokenSpan.getEndTokenIndex();
			DependencyParse dependencyParse = tokenSpan.getDocument().getDependencyParse(tokenSpan.getSentenceIndex());
			
			for (int i = startIndex; i < endIndex; i++) {
				if (this.mode == FeatureNGramDep.Mode.ChildrenOnly || this.mode == FeatureNGramDep.Mode.ParentsAndChildren) {
					List<DependencyParse.Dependency> dependencies = dependencyParse.getGovernedDependencies(i);
					for (DependencyParse.Dependency dependency : dependencies) {
						int depIndex = dependency.getDependentTokenIndex();
						if (depIndex <= document.getSentenceTokenCount(tokenSpan.getSentenceIndex()) - this.n 
								&& (depIndex < startIndex || depIndex >= endIndex)) {
							List<String> ngrams = getCleanNGramsAtPosition(document, tokenSpan.getSentenceIndex(), depIndex);
							for (String ngram : ngrams) {
								String retNgram = ngram + "_C";
								if (this.useRelationTypes)
									retNgram += "_" + ((dependency.getType() == null) ? "" : dependency.getType());
								
								if (!retNgrams.containsKey(retNgram))
									retNgrams.put(retNgram, 1);
								else
									retNgrams.put(ngram, retNgrams.get(retNgram) + 1);
							}
						}
					}
				}
				
				if (this.mode == FeatureNGramDep.Mode.ParentsOnly || this.mode == FeatureNGramDep.Mode.ParentsAndChildren) {
					List<DependencyParse.Dependency> dependencies = dependencyParse.getGoverningDependencies(i);
					for (DependencyParse.Dependency dependency : dependencies) {
						int govIndex = dependency.getGoverningTokenIndex();
						if (govIndex >= 0 && govIndex <= document.getSentenceTokenCount(tokenSpan.getSentenceIndex()) - this.n && (govIndex < startIndex || govIndex >= endIndex)) {
							List<String> ngrams = getCleanNGramsAtPosition(document, tokenSpan.getSentenceIndex(), govIndex);
							for (String ngram : ngrams) {
								String retNgram = ngram + "_P";
								if (this.useRelationTypes)
									retNgram += "_" + ((dependency.getType() == null) ? "" : dependency.getType());
								
								if (!retNgrams.containsKey(retNgram))
									retNgrams.put(retNgram, 1);
								else
									retNgrams.put(ngram, retNgrams.get(retNgram) + 1);
							}
						}
					}
				}
			}
		}
		
		return retNgrams;
	}

	@Override
	public String getGenericName() {
		return "NGramDep";
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureNGramDep<D, L>();
	}
	
	@Override
	public String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("mode"))
			return this.mode == null ? null : this.mode.toString();
		else if (parameter.equals("useRelationTypes"))
			return String.valueOf(this.useRelationTypes);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("mode"))
			this.mode = Mode.valueOf(parameterValue);
		else if (parameter.equals("useRelationTypes"))
			this.useRelationTypes = Boolean.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}

}
