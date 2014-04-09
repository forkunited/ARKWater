package ark.data.feature;

import java.util.*;

import ark.data.annotation.DataSet;
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
	public boolean init(DataSet<D, L> dataSet) {
		CounterTable<String> counter = new CounterTable<String>();
		for (D datum : dataSet) {
			Set<String> paths = getPathForDatum(datum);
			for (String path : paths) {
				counter.incrementCount(path);
			}
		}
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	private Set<String> getPathForDatum(D datum){
		
		Set<String> paths = new HashSet<String>();
		
		TokenSpan[] sourceTokenSpans = this.sourceTokenExtractor.extract(datum);
		TokenSpan[] targetTokenSpans = this.targetTokenExtractor.extract(datum);
		
		for (TokenSpan sourceSpan : sourceTokenSpans) {
			for (TokenSpan targetSpan : targetTokenSpans){
				if (sourceSpan.getSentenceIndex() != targetSpan.getSentenceIndex()){
					paths.add("NO_PATH");
					return paths;
				}
				paths.add(getShortestPath(sourceSpan, targetSpan));
			}
		}
		return paths;
	}
	
	private String getShortestPath(TokenSpan sourceSpan, TokenSpan targetSpan){
		// first loop over the tokens in the source and target
		// add each of the paths to a set
		// find the elemnet of the set with the shortest length
		List<DependencyPath> paths = new ArrayList<DependencyPath>();
		for (int i = sourceSpan.getStartTokenIndex(); i < sourceSpan.getEndTokenIndex(); i++){
			for (int j = targetSpan.getStartTokenIndex(); j < targetSpan.getEndTokenIndex(); j++){
				//System.out.println("num tokens in sent: " + sourceSpan.getDocument().getSentenceTokenCount(sourceSpan.getSentenceIndex()));
				if (sourceSpan.getDocument().getDependencyParse(sourceSpan.getSentenceIndex()).getPath(i, j) != null){
					paths.add(sourceSpan.getDocument().getDependencyParse(sourceSpan.getSentenceIndex()).getPath(i, j));
					
				}
			}
		}
		
		/*
		System.out.println("source start: " + sourceSpan.getStartTokenIndex());
		System.out.println("source end: " + sourceSpan.getEndTokenIndex());
		System.out.println("Source tokens: " + sourceSpan.toString());
		System.out.println("target start: " + targetSpan.getStartTokenIndex());
		System.out.println("target end: " + targetSpan.getEndTokenIndex());
		System.out.println("Target tokens: " + targetSpan.toString());
		System.out.println("num paths: " + paths.size());
		*/
		
		DependencyPath shortest = null;
		int shortestLen = Integer.MAX_VALUE;
		for (DependencyPath p : paths){
			if (p.tokenLength() < shortestLen){
				shortest = p;
				shortestLen = p.tokenLength();
			}
		}
		System.out.println("Shortest path: " + shortest);
		System.out.println();
		return shortest.toString();
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Set<String> prepsForDatum = getPathForDatum(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String prep : prepsForDatum) {
			if (this.vocabulary.containsKey(prep))
				vector.put(this.vocabulary.get(prep), 1.0);		
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

