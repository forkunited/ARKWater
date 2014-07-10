package ark.data.feature;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.ConstituencyParse.ConstituentPath;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;

/**
 * FeatureConstituencyPath computes paths in constituency parse trees
 * between token spans
 * associated with a datum. For a datum d with source token-span extractor S,
 * and target token span extractor T, the feature computes vector:
 * 
 * <1(p_1 \in P(S(d),T(d)), 1(p_2 \in P(S(d),T(d))), ... , 1(p_n \in P(S(d),T(d)))>
 * 
 * Where P(S(d),T(d)) gives the set of shortest constituency paths between token spans
 * in S(d) and token spans in T(d), and p_i is a constituency path in the vocabulary
 * of possible paths from the full data set containing d.
 *  
 * The 'minFeatureOccurrence' parameter determines the minimum number of times a
 * path p_i must appear in the full data set for it to have a component in the 
 * returned vectors.
 * 
 * The 'useRelationTypes' parameter determines whether the constituency paths corresponding
 * to components in the returned vector should be typed.
 * 
 * @author Jesse Dodge, Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureConstituencyPath<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	protected int minFeatureOccurrence;
	protected Datum.Tools.TokenSpanExtractor<D, L> sourceTokenExtractor;
	protected Datum.Tools.TokenSpanExtractor<D, L> targetTokenExtractor;
	protected boolean useRelationTypes = true;
	protected String[] parameterNames = {"minFeatureOccurrence", "sourceTokenExtractor", "targetTokenExtractor", "useRelationTypes"};
	
	public FeatureConstituencyPath(){
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
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
				ConstituentPath path = getShortestPath(sourceSpan, targetSpan);
				if (path == null)
					continue;
				paths.add(path.toString(this.useRelationTypes));
			}
		}
		return paths;
	}
	
	private ConstituentPath getShortestPath(TokenSpan sourceSpan, TokenSpan targetSpan){
		if (sourceSpan.getSentenceIndex() < 0 
				|| targetSpan.getSentenceIndex() < 0 
				|| sourceSpan.getSentenceIndex() != targetSpan.getSentenceIndex())
			return null;
		
		ConstituentPath shortestPath = null;
		int sentenceIndex = sourceSpan.getSentenceIndex();
		ConstituencyParse parse = sourceSpan.getDocument().getConstituencyParse(sentenceIndex);
		for (int i = sourceSpan.getStartTokenIndex(); i < sourceSpan.getEndTokenIndex(); i++){
			for (int j = targetSpan.getStartTokenIndex(); j < targetSpan.getEndTokenIndex(); j++){
				ConstituentPath path = parse.getPath(i, j);
				if (shortestPath == null || (path != null && path.getLength() < shortestPath.getLength()))
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
		return "ConstituencyPath";
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
		else if (parameter.equals("sourceTokenExtractor"))
			return (this.sourceTokenExtractor == null) ? null : this.sourceTokenExtractor.toString();
		else if (parameter.equals("targetTokenExtractor"))
			return (this.targetTokenExtractor == null) ? null : this.targetTokenExtractor.toString();
		else if (parameter.equals("useRelationTypes"))
			return String.valueOf(this.useRelationTypes);
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
		else if (parameter.equals("useRelationTypes"))
			this.useRelationTypes = Boolean.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureConstituencyPath<D, L>();
	}
}
