package ark.data.feature;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.feature.fn.Fn;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.BidirectionalLookupTable;

public class FeatureTokenSpanFnFilteredVocab<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	
	public enum VocabFilterType {
		SUFFIX,
		PREFIX,
		SUFFIX_OR_PREFIX,
		EQUAL
	}
	
	protected FeatureTokenSpanFnDataVocabTrie<D, L> vocabFeature;
	protected String vocabFilter;
	protected VocabFilterType vocabFilterType;
	protected Fn<List<TokenSpan>, List<String>> fn;
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected String[] parameterNames = {"vocabFeature", "vocabFilter", "vocabFilterType", "fn", "tokenExtractor"};
	
	private Fn.CacheMode fnCacheMode = Fn.CacheMode.NONE;
	
	public FeatureTokenSpanFnFilteredVocab() {
		
	}
	
	public FeatureTokenSpanFnFilteredVocab(Context<D, L> context) {
		this.context = context;
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
	}
	
	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("vocabFeature"))
			return this.vocabFeature.toParse();
		else if (parameter.equals("vocabFilter"))
			return Obj.stringValue(this.vocabFilter);
		else if (parameter.equals("vocabFilterType"))
			return Obj.stringValue(this.vocabFilterType.toString());
		else if (parameter.equals("fn"))
			return this.fn.toParse();
		else if (parameter.equals("tokenExtractor"))
			return Obj.stringValue(this.tokenExtractor.toString());
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("vocabFeature"))
			this.vocabFeature = (FeatureTokenSpanFnDataVocabTrie<D, L>)this.context.getMatchFeature(parameterValue);
		else if (parameter.equals("vocabFilter"))
			this.vocabFilter = this.context.getMatchValue(parameterValue);
		else if (parameter.equals("vocabFilterType"))
			this.vocabFilterType = VocabFilterType.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("fn"))
			this.fn = this.context.getMatchOrConstructTokenSpanStrFn(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = this.context.getDatumTools().getTokenSpanExtractor(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		Set<String> vocab = null;
		if (this.vocabFilterType == VocabFilterType.SUFFIX) {
			vocab = this.vocabFeature.getVocabularyTermsSuffixedBy(this.vocabFilter);
		} else if (this.vocabFilterType == VocabFilterType.PREFIX) {
			vocab = this.vocabFeature.getVocabularyTermsPrefixedBy(this.vocabFilter);
		} else if (this.vocabFilterType == VocabFilterType.SUFFIX_OR_PREFIX) {
			vocab = this.vocabFeature.getVocabularyTermsSuffixedBy(this.vocabFilter);
			vocab.addAll(this.vocabFeature.getVocabularyTermsPrefixedBy(this.vocabFilter));
		} else {
			vocab = new TreeSet<String>();
			if (this.vocabFeature.getVocabularyIndex(this.vocabFilter) != null)
				vocab.add(this.vocabFilter);
		}
		
		int i = 0;
		for (String term : vocab) {
			this.vocabulary.put(term, i);
			i++;
		}
		
		return true;
	}

	public Map<String, Integer> applyFnsToDatum(D datum) {
		List<TokenSpan> spans = Arrays.asList(this.tokenExtractor.extract(datum));
		List<String> strs = this.fn.compute(spans);

		Map<String, Integer> results = new HashMap<String, Integer>();
		
		for (String str : strs) {
			if (!results.containsKey(str))
				results.put(str, 0);
			results.put(str, results.get(str) + 1);
		}
		
		return results;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, int offset, Map<Integer, Double> vector) {
		List<TokenSpan> spans = Arrays.asList(this.tokenExtractor.extract(datum));
		List<String> strs = this.fn.compute(spans, this.tokenExtractor.toString() + datum.getId(), this.fnCacheMode);
		for (String str : strs) {
			if (this.vocabulary.containsKey(str))
				vector.put(this.vocabulary.get(str) + offset, 1.0);
		}
		
		return vector;
	}

	@Override
	public int getVocabularySize() {
		return this.vocabulary.size();
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
	protected <T extends Datum<Boolean>> Feature<T, Boolean> makeBinaryHelper(
			Context<T, Boolean> context, LabelIndicator<L> labelIndicator,
			Feature<T, Boolean> binaryFeature) {
		
		FeatureTokenSpanFnFilteredVocab<T, Boolean> binaryFeatureTokenSpanFnFilteredVocab = (FeatureTokenSpanFnFilteredVocab<T, Boolean>)binaryFeature;
		
		binaryFeatureTokenSpanFnFilteredVocab.vocabulary = this.vocabulary;
		
		return binaryFeatureTokenSpanFnFilteredVocab;
	}

	@Override
	protected boolean fromParseInternalHelper(AssignmentList internalAssignments) {
		return true;
	}

	@Override
	protected AssignmentList toParseInternalHelper(
			AssignmentList internalAssignments) {
		return internalAssignments;
	}

	@Override
	public Feature<D, L> makeInstance(Context<D, L> context) {
		return new FeatureTokenSpanFnFilteredVocab<D, L>(context);
	}

	@Override
	public String getGenericName() {
		return "TokenSpanFnFilteredVocab";
	}
	
	@Override
	protected boolean cloneHelper(Feature<D, L> clone) {
		FeatureTokenSpanFnFilteredVocab<D, L> cloneFilt = (FeatureTokenSpanFnFilteredVocab<D, L>)clone;
		cloneFilt.vocabulary = this.vocabulary;
		return true;
	}

	public void clearFnCaches() {
		this.fn.clearCaches();
	}
	
	public void setFnCacheMode(Fn.CacheMode fnCacheMode) {
		this.fnCacheMode = fnCacheMode;
	}
	
	public Fn<List<TokenSpan>, List<String>> getFn() {
		return this.fn;
	}
}
