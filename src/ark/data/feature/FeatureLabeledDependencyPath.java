package ark.data.feature;

import java.util.*;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.DependencyParse.DependencyPath;
import ark.data.annotation.nlp.TokenSpan;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

public class FeatureLabeledDependencyPath<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected Datum.Tools.TokenSpanExtractor<D, L> sourceTokenExtractor;
	protected Datum.Tools.TokenSpanExtractor<D, L> targetTokenExtractor;
	protected String[] parameterNames = {"minFeatureOccurrence", "sourceTokenExtractor", "targetTokenExtractor"};
	
	public FeatureLabeledDependencyPath(){
		vocabulary = new BidirectionalLookupTable<String, Integer>();
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Set<String> paths = getPathsForDatum(datum);
			for (String path : paths) {
				counter.incrementCount(path);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	private Set<String> getPathsForDatum(D datum){
		Set<String> paths = new HashSet<String>();
		
		TokenSpan[] sourceTokenSpans = this.sourceTokenExtractor.extract(datum);
		TokenSpan[] targetTokenSpans = this.targetTokenExtractor.extract(datum);
		
		for (TokenSpan sourceSpan : sourceTokenSpans) {
			for (TokenSpan targetSpan : targetTokenSpans){
				DependencyPath path = getShortestPath(sourceSpan, targetSpan);
				if (path == null)
					continue;
				paths.add(path.toString());
			}
		}
		return paths;
	}
	
	private DependencyPath getShortestPath(TokenSpan sourceSpan, TokenSpan targetSpan){
		if (sourceSpan.getSentenceIndex() < 0 
				|| targetSpan.getSentenceIndex() < 0 
				|| sourceSpan.getSentenceIndex() != targetSpan.getSentenceIndex())
			return null;
		
		DependencyPath shortestPath = null;
		int sentenceIndex = sourceSpan.getSentenceIndex();
		for (int i = sourceSpan.getStartTokenIndex(); i < sourceSpan.getEndTokenIndex(); i++){
			for (int j = targetSpan.getStartTokenIndex(); j < targetSpan.getEndTokenIndex(); j++){
				DependencyPath path = sourceSpan.getDocument().getDependencyParse(sentenceIndex).getPath(i, j);
				if (path != null && shortestPath != null && path.getTokenLength() < shortestPath.getTokenLength())
					shortestPath = path;
			}
		}

		return shortestPath;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> pathsForDatum = getPathsForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String path : pathsForDatum) {
			if (this.vocabulary.containsKey(path))
				vector.put(this.vocabulary.get(path), 1.0);		
		}

		return vector;
	}


	@Override
	public String getGenericName() {
		return "LabeledDependencyPath";
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
		else if (parameter.equals("sourceTokenExtractor"))
			return (sourceTokenExtractor == null) ? null : sourceTokenExtractor.toString();
		else if (parameter.equals("targetTokenExtractor"))
			return (targetTokenExtractor == null) ? null : targetTokenExtractor.toString();
		return null;
	}
	
	// note these will be called by TLinkDatum.Tools, and in that class TargetTokenSpan exists, for example.
	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence")) 
			this.minFeatureOccurrence = Integer.valueOf(parameterValue);
		else if (parameter.equals("sourceTokenExtractor"))
			this.sourceTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else if (parameter.equals("targetTokenExtractor"))
			this.targetTokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureLabeledDependencyPath<D, L>();
	}


}

