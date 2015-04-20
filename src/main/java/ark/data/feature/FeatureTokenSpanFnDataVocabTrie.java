package ark.data.feature;

import java.util.Map;
import java.util.Map.Entry;

import org.ardverk.collection.PatriciaTrie;
import org.ardverk.collection.StringKeyAnalyzer;
import org.ardverk.collection.Trie;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.parse.AssignmentList;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;
import ark.util.ThreadMapper;

public class FeatureTokenSpanFnDataVocabTrie<D extends Datum<L>, L> extends FeatureTokenSpanFnDataVocab<D, L> {
	private Trie<String, Double> forwardTrie;
	private Trie<String, Double> backwardTrie;
	
	public FeatureTokenSpanFnDataVocabTrie() {
		super();
	}
	
	public FeatureTokenSpanFnDataVocabTrie(Context<D, L> context) {
		super(context);
		
		this.forwardTrie = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
		this.backwardTrie = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		final CounterTable<String> counter = new CounterTable<String>();
		dataSet.map(new ThreadMapper.Fn<D, Boolean>() {
			@Override
			public Boolean apply(D datum) {
				Map<String, Integer> gramsForDatum = applyFnToDatum(datum);
				for (String gram : gramsForDatum.keySet()) {
					counter.incrementCount(gram);
				}
				return true;
			}
		});
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		Map<String, Integer> counts = counter.getCounts();
		double N = dataSet.size();
		for (Entry<String, Integer> entry : counts.entrySet()) {
			int id = this.vocabulary.get(entry.getKey());
			double idf = Math.log(N/(1.0 + entry.getValue()));
			this.idfs.put(id, idf);
			this.forwardTrie.put(entry.getKey(), idf);
			this.backwardTrie.put(new StringBuilder(entry.getKey()).reverse().toString(), idf);
		}
		
		return true;
	}
	
	@Override
	protected <T extends Datum<Boolean>> Feature<T, Boolean> makeBinaryHelper(
			Context<T, Boolean> context, LabelIndicator<L> labelIndicator,
			Feature<T, Boolean> binaryFeature) {
		FeatureTokenSpanFnDataVocabTrie<T, Boolean> binaryFeatureTokenSpanFnDataVocabTrie = (FeatureTokenSpanFnDataVocabTrie<T, Boolean>)super.makeBinaryHelper(context, labelIndicator, binaryFeature);
		binaryFeatureTokenSpanFnDataVocabTrie.forwardTrie = this.forwardTrie;
		binaryFeatureTokenSpanFnDataVocabTrie.backwardTrie = this.backwardTrie;
		
		
		return binaryFeatureTokenSpanFnDataVocabTrie;
	}

	@Override
	protected boolean fromParseInternalHelper(AssignmentList internalAssignments) {
		if (!super.fromParseInternalHelper(internalAssignments))
			return false;
		if (internalAssignments == null)
			return true;
	
		this.forwardTrie = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
		this.backwardTrie = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
		
		for (int i = 0; i < this.vocabulary.size(); i++) {
			String term = this.vocabulary.reverseGet(i);
			this.forwardTrie.put(term, this.idfs.get(i));
			this.backwardTrie.put(new StringBuilder(term).reverse().toString(), this.idfs.get(i));
		}
		
		return true;
	}

	@Override
	public Feature<D, L> makeInstance(Context<D, L> context) {
		return new FeatureTokenSpanFnDataVocabTrie<D, L>(context);	
	}

	@Override
	public String getGenericName() {
		return "TokenSpanFnDataVocabTrie";
	}
	
	public Map<String, Double> getVocabularyTermsPrefixedBy(String prefix) {
		return this.forwardTrie.prefixMap(prefix);
	}
	
	public Map<String, Double> getVocabularyTermsSuffixedBy(String suffix) {
		return this.backwardTrie.prefixMap(suffix);
	}	
	
	public double getVocabularyTermIdf(String term) {
		if (this.vocabulary.containsKey(term))
			return this.idfs.get(this.vocabulary.get(term));
		else
			return -1.0;
	}
}
