package ark.data.feature;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

public class FeatureUnlabeledDependencyPath<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected String POS;
	protected String[] parameterNames = {"minFeatureOccurrence", "tokenExtractor", "POS"};
	
	public FeatureUnlabeledDependencyPath(){
		vocabulary = new BidirectionalLookupTable<String, Integer>();
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Set<String> preps = getPrepsForDatum(datum);
			for (String ngram : preps) {
				counter.incrementCount(ngram);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	private Set<String> getPrepsForDatum(D datum){
		Set<String> preps = new HashSet<String>();
		
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		
		for (TokenSpan tokenSpan : tokenSpans) {
			
			//List<String> tokens = tokenSpan.getDocument().getSentenceTokens(tokenSpan.getSentenceIndex());
			int sentLength = tokenSpan.getDocument().getSentenceTokenCount(tokenSpan.getSentenceIndex());
			
			int sentIndex = tokenSpan.getSentenceIndex();
			
			for (int tokenIndex = 0 ; tokenIndex < sentLength; tokenIndex++){
				if (tokenSpan.getDocument().getPoSTag(sentIndex, tokenIndex).toString().equals(POS)){
					preps.add(tokenSpan.getDocument().getToken(sentIndex, tokenIndex));
				}
				
			}
		}
		
		return preps;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> prepsForDatum = getPrepsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String prep : prepsForDatum) {
			if (this.vocabulary.containsKey(prep))
				vector.put(this.vocabulary.get(prep), 1.0);		
		}

		return vector;
	}


	@Override
	public String getGenericName() {
		return "Prepositions";
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
		else if (parameter.equals("tokenExtractor"))
			return (this.tokenExtractor == null) ? null : this.tokenExtractor.toString();
		else if (parameter.equals("POS"))
			return this.POS;
		return null;
	}
	
	// note these will be called by TLinkDatum.Tools, and in that class TargetTokenSpan exists, for example.
	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("POS"))
			this.POS = parameterValue;
		else
			return false;
		return true;
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureUnlabeledDependencyPath<D, L>();
	}


}

