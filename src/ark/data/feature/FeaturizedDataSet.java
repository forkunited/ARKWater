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

import org.platanios.learn.data.DataSetInMemory;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.util.ThreadMapper;
import ark.util.ThreadMapper.Fn;

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
	
	private List<Feature<D, L>> featureList; // Just to keep all of the features referenced in one place for when cloning the dataset
	private Map<String, Feature<D, L>> referencedFeatures; // Maps from reference names to features
	private TreeMap<Integer, Feature<D, L>> features; // Maps from the feature's starting vocabulary index to the feature
	private Map<Integer, String> featureVocabularyNames; // Sparse map from indices to names
	private Map<Integer, Vector> featureVocabularyValues; // Map from datum ids to indices to values
	private int featureVocabularySize;
	private boolean precomputedFeatures;
	
	private DataSetInMemory<PredictedDataInstance<Vector, Integer>> plataniosDataSet;

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
		 
		this.featureList = new ArrayList<Feature<D, L>>();
		this.featureVocabularySize = 0;
		for (Feature<D, L> feature : features)
			addFeature(feature);
		
		this.featureVocabularyNames = new ConcurrentHashMap<Integer, String>();
		this.featureVocabularyValues = new ConcurrentHashMap<Integer, Vector>();
		this.precomputedFeatures = false;
	}
	
	public String getName() {
		return this.name;
	}
	
	public int getMaxThreads() {
		return this.maxThreads;
	}
	
	public boolean setMaxThreads(int maxThreads) {
		this.maxThreads = maxThreads;
		return true;
	}
	
	public <T> List<T> map(final ThreadMapper.Fn<D, T> fn) {
		return map(fn, this.maxThreads);
	}
	
	/**
	 * @param feature
	 * @param initFeature
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
	public boolean addFeature(Feature<D, L> feature, boolean initFeature) {
		if (initFeature)
			if (!feature.init(this))
				return false;
		return addFeatureHelper(feature);
	}
	
	public boolean addFeature(Feature<D, L> feature) {
		return addFeature(feature, false);
	}
	
	public boolean addFeatures(List<Feature<D, L>> features, boolean initFeatures) {
		if (initFeatures) {
			final FeaturizedDataSet<D, L> data = this;
			ThreadMapper<Feature<D, L>, Boolean> threads = new ThreadMapper<Feature<D, L>, Boolean>(new Fn<Feature<D, L>, Boolean>() {
				public Boolean apply(Feature<D, L> feature) {
					return feature.init(data);
				}
			});
			
			List<Boolean> threadResults = threads.run(features, this.maxThreads);
			for (boolean threadResult : threadResults)
				if (!threadResult)
					return false;
		}
		
		for (Feature<D, L> feature : features)
			if (!addFeatureHelper(feature))
				return false;
		
		return true;
	}
	
	public boolean addFeatures(List<Feature<D, L>> features) {
		return addFeatures(features, false);
	}
	
	
	private boolean addFeatureHelper(Feature<D, L> feature) {
		if (!feature.isIgnored()) {
			this.features.put(this.featureVocabularySize, feature);
			this.featureVocabularySize += feature.getVocabularySize();
		}
		if (feature.getReferenceName() != null)
			this.referencedFeatures.put(feature.getReferenceName(), feature);
		this.featureList.add(feature);
		
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
			featureVocabularyNames.addAll(feature.getSpecificShortNames()); 
		}
		
		return featureVocabularyNames;
	}
	
	public Map<Integer, Double> getFeatureVocabularyValuesAsMap(D datum) {
		// FIXME Make this faster.
		Map<Integer, Double> map = new HashMap<Integer, Double>();
		Vector vector = getFeatureVocabularyValues(datum);
		double[] denseArray = vector.getDenseArray();
		for (int i = 0; i < denseArray.length; i++)
			if (denseArray[i] > 0)
				map.put(i, denseArray[i]);
		return map;
	}
	
	public Vector getFeatureVocabularyValues(D datum) {
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
		
		Vector vector = new SparseVector(getFeatureVocabularySize(), values);
		
		this.featureVocabularyValues.put(datum.getId(), vector);
		
		return vector;
	}
	
	public boolean precomputeFeatures() {
		if (this.precomputedFeatures)
			return true;
		
		List<Boolean> threadResults = map(new ThreadMapper.Fn<D, Boolean>() {
			@Override
			public Boolean apply(D datum) {
				Vector featureVector = getFeatureVocabularyValues(datum);
					if (featureVector == null)
						return false;
				
				return true;
			}
		}, this.maxThreads);
		

		for (boolean result : threadResults)
			if (!result)
				return false;
		
		this.precomputedFeatures = true;
		return true;
	}
	
	@Override
	public <T extends Datum<Boolean>> DataSet<T, Boolean> makeBinaryDataSet(String labelIndicator, Datum.Tools<T, Boolean> datumTools) {
		List<Feature<T, Boolean>> features = new ArrayList<Feature<T, Boolean>>();
		for (Feature<D, L> feature : this.featureList) {
			features.add(feature.clone(datumTools, datumTools.getDataTools().getParameterEnvironment(), false));
		}
		
		FeaturizedDataSet<T, Boolean> dataSet = new FeaturizedDataSet<T, Boolean>(this.name + ((labelIndicator == null) ? "" : "/" + labelIndicator), features, 1, datumTools, null);
		LabelIndicator<L> indicator = (labelIndicator == null) ? null : getDatumTools().getLabelIndicator(labelIndicator);
		
		for (D datum : this) {
			dataSet.add(getDatumTools().<T>makeBinaryDatum(datum, indicator));
		}
		
		dataSet.featureVocabularySize = this.featureVocabularySize;
		dataSet.featureVocabularyNames = this.featureVocabularyNames;
		dataSet.featureVocabularyValues = this.featureVocabularyValues;
		
		return dataSet;
	}
	
	public synchronized DataSetInMemory<PredictedDataInstance<Vector, Integer>> makePlataniosDataSet() {
		if (this.plataniosDataSet != null)
			return this.plataniosDataSet;
		
		ThreadMapper<D, PredictedDataInstance<Vector, Integer>> threadMapper 
			= new ThreadMapper<D, PredictedDataInstance<Vector, Integer>>(new ThreadMapper.Fn<D, PredictedDataInstance<Vector, Integer>>() {
				@Override
				public PredictedDataInstance<Vector, Integer> apply(D datum) {
					Vector vector = getFeatureVocabularyValues(datum);
					Integer label = null;
					if (datum.getLabel() != null)
						label = (Boolean)datum.getLabel() ? 1 : 0;
				
					return new PredictedDataInstance<Vector, Integer>(String.valueOf(datum.getId()), vector, label, null, 1);
				}
			});
		
		List<PredictedDataInstance<Vector, Integer>> dataInstances = threadMapper.run(new ArrayList<D>(this.data.values()), this.maxThreads);
		this.plataniosDataSet = new DataSetInMemory<PredictedDataInstance<Vector, Integer>>(dataInstances);
		
		return this.plataniosDataSet;
	}
}
