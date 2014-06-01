package ark.data.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.util.CounterTable;
import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.BidirectionalLookupTable;
import ark.wrapper.BrownClusterer;

public abstract class FeatureNGram<D extends Datum<L>, L> extends Feature<D, L> {
	public enum Scale {
		INDICATOR,
		NORMALIZED_LOG
	}
	
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	protected Map<Integer, Double> idfs;
	
	protected int minFeatureOccurrence;
	protected int n;
	protected DataTools.StringTransform cleanFn;
	protected BrownClusterer clusterer;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected Scale scale;
	protected String[] parameterNames = {"minFeatureOccurrence", "n", "cleanFn", "clusterer", "tokenExtractor", "scale"};
	
	
	protected abstract Map<String, Integer> getNGramsForDatum(D datum);
	
	public FeatureNGram() {
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
		this.idfs = new HashMap<Integer, Double>();
		this.scale = Scale.INDICATOR;
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Map<String, Integer> ngramsForDatum = getNGramsForDatum(datum);
			for (String ngram : ngramsForDatum.keySet()) {
				counter.incrementCount(ngram);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<String, Integer> ngramsForDatum = getNGramsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		if (this.scale == Scale.INDICATOR) {
			for (String ngram : ngramsForDatum.keySet()) {
				if (this.vocabulary.containsKey(ngram))
					vector.put(this.vocabulary.get(ngram), 1.0);		
			}
		} else if (this.scale == Scale.NORMALIZED_LOG) {
			double norm = 0.0;
			for (Entry<String, Integer> entry : ngramsForDatum.entrySet()) {
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
		}

		return vector;
	}

	protected List<String> getCleanNGrams(List<String> tokens, int startIndex) {
		List<String> ngram = new ArrayList<String>(this.n);
		for (int i = startIndex; i < startIndex + this.n; i++)
			ngram.add(tokens.get(i));
		
		List<String> retNgrams = new ArrayList<String>();
		if (this.n == 1 && this.clusterer != null) {
			String cluster = this.clusterer.getCluster(this.cleanFn.transform(ngram.get(0)));
			if (cluster == null)
				return null;
			for (int i = 2; i < cluster.length(); i *= 2) {
				retNgrams.add(cluster.substring(0, i));
			}
			return retNgrams;
		}
		
		StringBuilder ngramGlue = new StringBuilder();
		for (String gram : ngram) {
			String cleanGram = this.cleanFn.transform(gram);
			if (cleanGram.length() == 0)
				continue;
			
			if (this.clusterer != null) {
				String cluster = this.clusterer.getCluster(cleanGram);
				if (cluster != null) {
					ngramGlue = ngramGlue.append(cluster).append("_");
				} 
			} else { 
				ngramGlue = ngramGlue.append(cleanGram).append("_");
			}
		}
		
		if (ngramGlue.length() == 0)
			return null;
		
		ngramGlue = ngramGlue.delete(ngramGlue.length() - 1, ngramGlue.length());
		retNgrams.add(ngramGlue.toString());
		
		return retNgrams;
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
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("minFeatureOccurrence")) 
			return String.valueOf(this.minFeatureOccurrence);
		else if (parameter.equals("n"))
			return String.valueOf(this.n);
		else if (parameter.equals("cleanFn"))
			return (this.cleanFn == null) ? null : this.cleanFn.toString();
		else if (parameter.equals("clusterer"))
			return (this.clusterer == null) ? "None" : this.clusterer.getName();
		else if (parameter.equals("tokenExtractor"))
			return (this.tokenExtractor == null) ? null : this.tokenExtractor.toString();
		else if (parameter.equals("scale"))
			return this.scale.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("n"))
			this.n = Integer.valueOf(parameterValue);
		else if (parameter.equals("cleanFn"))
			this.cleanFn = datumTools.getDataTools().getCleanFn(parameterValue);
		else if (parameter.equals("clusterer"))
			this.clusterer = datumTools.getDataTools().getBrownClusterer(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("scale"))
			this.scale = Scale.valueOf(parameterValue);
		else
			return false;
		return true;
	}
}
