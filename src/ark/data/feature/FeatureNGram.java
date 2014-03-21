package ark.data.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ark.util.CounterTable;
import ark.data.DataTools;
import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.util.BidirectionalLookupTable;
import ark.util.Stemmer;
import ark.wrapper.BrownClusterer;

public abstract class FeatureNGram<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected int n;
	protected DataTools.StringTransform cleanFn;
	protected BrownClusterer clusterer;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected String[] parameterNames = {"minFeatureOccurrence", "n", "cleanFn", "clusterer", "tokenExtractor"};
	
	
	protected abstract Set<String> getNGramsForDatum(D datum);
	
	@Override
	public boolean init(DataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Set<String> ngramsForDatum = getNGramsForDatum(datum);
			for (String ngram : ngramsForDatum) {
				counter.incrementCount(ngram);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> ngramsForDatum = getNGramsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String ngram : ngramsForDatum) {
			if (this.vocabulary.containsKey(ngram))
				vector.put(this.vocabulary.get(ngram), 1.0);		
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
				ngramGlue = ngramGlue.append(Stemmer.stem(cleanGram)).append("_");
			}
		}
		
		if (ngramGlue.length() == 0)
			return null;
		
		ngramGlue = ngramGlue.delete(ngramGlue.length() - 1, ngramGlue.length());
		retNgrams.add(ngramGlue.toString());
		
		return retNgrams;
	}

	@Override
	protected String getVocabularyTerm(int index) {
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
			return this.cleanFn.toString();
		else if (parameter.equals("clusterer"))
			return (this.clusterer == null) ? "None" : this.clusterer.getName();
		else if (parameter.equals("tokenExtractor"))
			return this.tokenExtractor.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, DataTools dataTools, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("n"))
			this.n = Integer.valueOf(parameterValue);
		else if (parameter.equals("cleanFn"))
			this.cleanFn = dataTools.getCleanFn(parameterValue);
		else if (parameter.equals("clusterer"))
			this.clusterer = dataTools.getBrownClusterer(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else
			return false;
		return true;
	}
}
