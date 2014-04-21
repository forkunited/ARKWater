package ark.data.feature;

import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.nlp.TokenSpan;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

public class FeatureSurfaceDistance <D extends Datum<L>, L> extends Feature<D, L>{
	
	protected BidirectionalLookupTable<String, Integer> vocabulary;

	protected Datum.Tools.TokenSpanExtractor<D, L> sourceTokenExtractor;
	protected Datum.Tools.TokenSpanExtractor<D, L> targetTokenExtractor;
	protected String[] parameterNames = {"sourceTokenExtractor", "targetTokenExtractor", "PoS"};
	
	public FeatureSurfaceDistance(){
		vocabulary = new BidirectionalLookupTable<String, Integer>();
	}

	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		
		for (D datum : dataSet) {
			counter.incrementCount(findDistance(datum));
		}
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		return true;
	}
	
	private String findDistance(D datum){
		TokenSpan source = sourceTokenExtractor.extract(datum)[0];
		TokenSpan target = targetTokenExtractor.extract(datum)[0];
		return "" + (target.getStartTokenIndex() - source.getStartTokenIndex());
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<Integer, Double> vect = new HashMap<Integer, Double>();
		vect.put(vocabulary.get(findDistance(datum)), 1.0);
		return vect;
	}

	@Override
	public String getGenericName() {
		return "surfaceDistance";
	}

	@Override
	public int getVocabularySize() {
		return vocabulary.size();
	}

	@Override
	public String getVocabularyTerm(int index) {
		return "" + vocabulary.reverseGet(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		vocabulary.put(term, index);
		return true;
	}

	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("sourceTokenExtractor")) 
			return (this.sourceTokenExtractor == null) ? null : this.sourceTokenExtractor.toString();
		else if (parameter.equals("targetTokenExtractor"))
			return (this.targetTokenExtractor == null) ? null : this.targetTokenExtractor.toString();
		return null;	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("sourceTokenExtractor")) 
			this.sourceTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("targetTokenExtractor")) 
			this.targetTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else
			return false;
		return true;

	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureSurfaceDistance<D,L>();
	}

}
