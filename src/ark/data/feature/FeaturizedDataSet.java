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

public class FeaturizedDataSet<D extends Datum<L>, L> extends DataSet<D, L> {
	private String name;
	private int maxThreads;
	
	private Map<String, Integer> referencedFeatures; // Maps from reference names to features
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
		this.referencedFeatures = new HashMap<String, Integer>();
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
	
	public boolean addFeature(Feature<D, L> feature) {
		this.features.put(this.featureVocabularySize, feature);
		
		if (feature.getReferenceName() != null)
			this.referencedFeatures.put(feature.getReferenceName(), this.featureVocabularySize);
		
		this.featureVocabularySize += feature.getVocabularySize();
		
		return true;
	}
	
	public Feature<D, L> getFeature(int index) {
		return this.features.get(index);
	}
	
	public Feature<D, L> getFeatureByReferenceName(String referenceName) {
		return getFeature(this.referencedFeatures.get(referenceName));
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
				featuresToIndices.get(featureIndex).add((featureIndex != 0) ? index % featureIndex : index);
			}
		}
		
		for (Entry<Integer, List<Integer>> featureToIndices : featuresToIndices.entrySet()) {
			Feature<D, L> feature = this.features.get(featureToIndices.getKey());
			Map<Integer, String> featureSpecificNames = feature.getSpecificShortNamesForIndices(featureToIndices.getValue());
			for (Entry<Integer, String> featureSpecificIndexToName : featureSpecificNames.entrySet())
				names.put(featureSpecificIndexToName.getKey() + featureToIndices.getKey(), featureSpecificIndexToName.getValue());
		}
		
		return names;
	}
	
	public List<String> getFeatureVocabularyNames() {
		List<String> featureVocabularyNames = new ArrayList<String>(this.featureVocabularySize);
		
		for (Feature<D, L> feature : this.features.values()) {
			featureVocabularyNames.addAll(feature.getSpecificShortNames());
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
