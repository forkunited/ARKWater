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
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.TreeMap;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;

/**
 * DataSet represents a collection of labeled and/or unlabeled 'datums'
 * that have been featurized by a provided set of features to train and 
 * evaluate models.  
 * 
 * The current implementation computes the features on demand as their
 * values are requested, and permanently caches their values in memory.
 * In the future, this might be improved so that some values can be
 * evicted from the cache and possibly serialized/deserialized from
 * disk.
 * 
 * @author Bill McDowell
 *
 * @param <D> Datum type
 * @param <L> Datum label type
 */
public class FeaturizedDataSet<D extends Datum<L>, L> extends DataSet<D, L> {
	private String name;
	private int maxThreads;
	
	private Map<String, Feature<D, L>> referencedFeatures; // Maps from reference names to features
	private TreeMap<Integer, Feature<D, L>> features; // Maps from the feature's starting vocabulary index to the feature
	private Map<Integer, String> featureVocabularyNames; // Sparse map from indices to names
	private Map<D, Map<Integer, Double>> featureVocabularyValues; // Map from data to indices to values
	private int featureVocabularySize;
	
	public FeaturizedDataSet(String name, Datum.Tools<D, L> datumTools, Datum.Tools.LabelMapping<L> labelMapping) {
		this(name, 1, datumTools, labelMapping);
	}
	
	public FeaturizedDataSet(String name, int maxThreads, Datum.Tools<D, L> datumTools, Datum.Tools.LabelMapping<L> labelMapping) {
		this(name, new ArrayList<Feature<D, L>>(), maxThreads, datumTools, labelMapping);
	}
	
	public FeaturizedDataSet(String name, List<Feature<D, L>> features, int maxThreads, Datum.Tools<D, L> datumTools, Datum.Tools.LabelMapping<L> labelMapping) {
		super(datumTools, labelMapping);
		this.name = name;
		this.referencedFeatures = new HashMap<String, Feature<D, L>>();
		this.features = new TreeMap<Integer, Feature<D, L>>();
		this.maxThreads = maxThreads;
		 
		this.featureVocabularySize = 0;
		for (Feature<D, L> feature : features)
			addFeature(feature);
		
		this.featureVocabularyNames = new ConcurrentHashMap<Integer, String>();
		this.featureVocabularyValues = new ConcurrentHashMap<D, Map<Integer, Double>>();
	}
	
	public String getName() {
		return this.name;
	}
	
	public int getMaxThreads() {
		return this.maxThreads;
	}
	
	/**
	 * @param feature
	 * @return true if the feature has been added.  This does *not* call
	 * the feature's initialization method, so it must be called outside
	 * the FeaturizedDataSet before the feature is added to the set (this
	 * is necessary so that the feature can sometimes be initialized using
	 * a different data set from the one to which it is added)
	 * 
	 * If the feature is ignored (feature.isIgnored returns true), then
	 * the feature's values will not be included in vectors returned by requests
	 * to this FeaturizedDataSet, and so it will not be used by models that
	 * used this FeaturizedDataSet.
	 * 
	 * Features can be retrieved from this by their 'reference names' (even
	 * if they are ignored features).  This is useful when one feature is computed
	 * using the values of another.  If one feature requires another for its
	 * computation, but the required feature should not be included in a model that
	 * refers to FeaturizedDataSet, then the required feature should be 
	 * set to be 'ignored'.
	 * 
	 */
	public boolean addFeature(Feature<D, L> feature) {
		if (!feature.isIgnored()) {
			this.features.put(this.featureVocabularySize, feature);
			this.featureVocabularySize += feature.getVocabularySize();
		}
		if (feature.getReferenceName() != null)
			this.referencedFeatures.put(feature.getReferenceName(), feature);
		
		return true;
	}
	
	public Feature<D, L> getFeature(int index) {
		return this.features.get(index);
	}
	
	public Feature<D, L> getFeatureByReferenceName(String referenceName) {
		return this.referencedFeatures.get(referenceName);
	}
	
	public int getFeatureCount() {
		return this.features.size();
	}
	
	public int getFeatureVocabularySize() {
		return this.featureVocabularySize;
	}
	
	public Map<Integer, String> getFeatureVocabularyNamesForIndices(Iterable<Integer> indices) {
		Map<Integer, String> names = new HashMap<Integer, String>();
		Map<Integer, List<Integer>> featuresToIndices = new HashMap<Integer, List<Integer>>();
		for (Integer index : indices) {
			if (this.featureVocabularyNames.containsKey(index)) {
				names.put(index, this.featureVocabularyNames.get(index));
			} else {
				Integer featureIndex = this.features.floorKey(index);
				if (!featuresToIndices.containsKey(featureIndex))
					featuresToIndices.put(featureIndex, new ArrayList<Integer>());
				featuresToIndices.get(featureIndex).add(index - featureIndex);
			}
		}
		
		for (Entry<Integer, List<Integer>> featureToIndices : featuresToIndices.entrySet()) {
			Feature<D, L> feature = this.features.get(featureToIndices.getKey());
			Map<Integer, String> featureSpecificNames = feature.getSpecificShortNamesForIndices(featureToIndices.getValue());
			for (Entry<Integer, String> featureSpecificIndexToName : featureSpecificNames.entrySet())
				names.put(featureSpecificIndexToName.getKey() + featureToIndices.getKey(), featureSpecificIndexToName.getValue());
		}
		
		this.featureVocabularyNames.putAll(names);
		
		return names;
	}
	
	public List<String> getFeatureVocabularyNames() {
		List<String> featureVocabularyNames = new ArrayList<String>(this.featureVocabularySize);
		
		for (Feature<D, L> feature : this.features.values()) {
			featureVocabularyNames.addAll(feature.getSpecificShortNames()); // FIXME Maybe this should cache...
		}
		
		return featureVocabularyNames;
	}
	
	public Map<Integer, Double> getFeatureVocabularyValues(D datum) {
		if (!this.data.containsKey(datum.getId()))
			return null;
		if (this.featureVocabularyValues.containsKey(datum.getId()))
			return this.featureVocabularyValues.get(datum.getId());
		
		Map<Integer, Double> values = new HashMap<Integer, Double>();
		for (Entry<Integer, Feature<D, L>> featureEntry : this.features.entrySet()) {
			Map<Integer, Double> featureValues = featureEntry.getValue().computeVector(datum);
			for (Entry<Integer, Double> featureValueEntry : featureValues.entrySet()) {
				values.put(featureValueEntry.getKey() + featureEntry.getKey(), featureValueEntry.getValue());
			}
		}
		
		this.featureVocabularyValues.put(datum, values);
		
		return values;
	}
	
	public boolean precomputeFeatures() {
		/* FIXME: Implement threaded. Also maybe implement serialization methods */
		return true;
	}
}
