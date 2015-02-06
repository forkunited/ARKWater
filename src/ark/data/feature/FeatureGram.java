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
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.util.CounterTable;
import ark.util.Pair;
import ark.util.SerializationUtil;
import ark.util.ThreadMapper;
import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.BidirectionalLookupTable;

/**
 * 
 * 
 * @author Bill McDowell
 * 
 * @param <D> datum type
 * @param <L> datum label type
 *
 */
public abstract class FeatureGram<D extends Datum<L>, L> extends Feature<D, L> {
	/**
	 * 
	 * Scale gives possible functions by which to scale the computed
	 * feature vectors.  The INDICATOR function just returns 1 if an n-gram is 
	 * in F(T(d)) for datum d (F and T defined in documenation above).  The
	 * NORMALIZED_LOG function applies log(1+tf(F(T(d)), v) to n-gram v, where
	 * tf(x,v) computes the frequency of v in x.   Similarly, NORMALIZED_TFIDF
	 * applies tfidf for each n-gram.  Both NORMALIZED_LOG and NORMALIZED_TFIDF
	 * are normalized in the sense that the feature vector for n-gram v is 
	 * scaled to length 1.
	 * 
	 */
	public enum Scale {
		INDICATOR,
		NORMALIZED_LOG,
		NORMALIZED_TFIDF
	}
	
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	protected Map<Integer, Double> idfs; // maps vocabulary term indices to idf values to use in tfidf scale function
	
	protected int minFeatureOccurrence;
	protected DataTools.StringTransform cleanFn;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected Scale scale;
	protected String[] parameterNames = {"minFeatureOccurrence", "cleanFn", "tokenExtractor", "scale"};
	
	/**
	 * @param datum
	 * @return grams associated with the datum in a certain way that
	 * depends on the the particular Gram feature that is being computed.  See
	 * ark.data.feature.FeatureNGram for example.
	 */
	protected abstract Map<String, Integer> getGramsForDatum(D datum);
	
	public FeatureGram() {
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
		this.idfs = new HashMap<Integer, Double>();
		this.scale = Scale.INDICATOR;
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		final CounterTable<String> counter = new CounterTable<String>();
		dataSet.map(new ThreadMapper.Fn<D, Boolean>() {
			@Override
			public Boolean apply(D datum) {
				Map<String, Integer> gramsForDatum = getGramsForDatum(datum);
				for (String gram : gramsForDatum.keySet()) {
					counter.incrementCount(gram);
				}
				return true;
			}
		});
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		Map<String, Integer> counts = counter.getCounts();
		double N = dataSet.size();
		for (Entry<String, Integer> entry : counts.entrySet()) {
			this.idfs.put(this.vocabulary.get(entry.getKey()), Math.log(N/(1.0 + entry.getValue())));
		}
		
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<String, Integer> gramsForDatum = getGramsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		if (this.scale == Scale.INDICATOR) {
			for (String gram : gramsForDatum.keySet()) {
				if (this.vocabulary.containsKey(gram))
					vector.put(this.vocabulary.get(gram), 1.0);		
			}
		} else if (this.scale == Scale.NORMALIZED_LOG) {
			double norm = 0.0;
			for (Entry<String, Integer> entry : gramsForDatum.entrySet()) {
				if (!this.vocabulary.containsKey(entry.getKey()))
					continue;
				int index = this.vocabulary.get(entry.getKey());
				double value = Math.log(entry.getValue() + 1.0);
				norm += value*value;
				vector.put(index, value);
			}
			
			norm = Math.sqrt(norm);
			
			for (Entry<Integer, Double> entry : vector.entrySet()) {
				entry.setValue(entry.getValue()/norm);
			}
		} else if (this.scale == Scale.NORMALIZED_TFIDF) {
			double norm = 0.0;
			for (Entry<String, Integer> entry : gramsForDatum.entrySet()) {
				if (!this.vocabulary.containsKey(entry.getKey()))
					continue;
				int index = this.vocabulary.get(entry.getKey());
				double value = entry.getValue()*this.idfs.get(index);//Math.log(entry.getValue() + 1.0);
				norm += value*value;
				vector.put(index, value);
			}
			
			norm = Math.sqrt(norm);
			
			for (Entry<Integer, Double> entry : vector.entrySet()) {
				entry.setValue(entry.getValue()/norm);
			}
		}

		return vector;
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
		else if (parameter.equals("cleanFn"))
			return (this.cleanFn == null) ? null : this.cleanFn.toString();
		else if (parameter.equals("tokenExtractor"))
			return (this.tokenExtractor == null) ? null : this.tokenExtractor.toString();
		else if (parameter.equals("scale"))
			return this.scale.toString();
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("cleanFn"))
			this.cleanFn = datumTools.getDataTools().getCleanFn(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("scale"))
			this.scale = Scale.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	protected <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects) {
		FeatureGram<D1,L1> cloneFeature = (FeatureGram<D1, L1>)clone;
		if (!newObjects) {
			cloneFeature.vocabulary = this.vocabulary;
			cloneFeature.idfs = this.idfs;
		} else {
			cloneFeature.idfs = new HashMap<Integer, Double>();
			for (Entry<Integer, Double> entry : this.idfs.entrySet())
				cloneFeature.idfs.put(entry.getKey(), entry.getValue());
		}
		
		return true;
	}
	
	@Override
	protected boolean serializeHelper(Writer writer) throws IOException {
		int i = 0;
		for (Entry<Integer, Double> idf : this.idfs.entrySet()) {
			Pair<String, Double> assign = new Pair<String, Double>(idf.getKey().toString(), idf.getValue());
			if (!SerializationUtil.serializeAssignment(assign, writer))
				return false;
			if (i != this.idfs.size() - 1)
				writer.write(",");
			i++;
		}
		
		return true;
	}
	
	@Override
	protected boolean deserializeHelper(BufferedReader reader) throws IOException {
		Map<String,String> idfStrs = SerializationUtil.deserializeArguments(reader);
		if (idfStrs == null)
			return false;
		
		this.idfs = new HashMap<Integer,Double>();
		for (Entry<String, String> entry : idfStrs.entrySet()) {
			this.idfs.put(Integer.valueOf(entry.getKey()), Double.valueOf(entry.getValue()));
		}
		
		return true;
	}
}
