package ark.data.feature;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

/**
 * For each datum d FeatureNGramSentence computes a
 * vector:
 * 
 * <c(v_1\in S(T(d))), c(v_2 \in S(T(d))), ... , c(v_n \in S(T(d)))>
 * 
 * Where T is a token extractor, S(T(d)) computes the n-grams 
 * in the sentence surrounding the tokens given by T(d) in a source text document,
 * and c(v \in S(T(d))) computes the number of occurrences of n-gram v in S.  
 * The resulting
 * vector is given to methods in ark.data.feature.FeatureNGram to be normalized
 * and scaled in some way.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureNGramSentence<D extends Datum<L>, L> extends FeatureNGram<D, L> {

	@Override
	protected Map<String, Integer> getNGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Map<String, Integer> retNgrams = new HashMap<String, Integer>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			if (tokenSpan.getSentenceIndex() < 0)
				continue;
			
			List<String> tokens = tokenSpan.getDocument().getSentenceTokens(tokenSpan.getSentenceIndex());
			if (tokens == null)
				continue;
			for (int i = 0; i < tokens.size()-this.n+1; i++) {
				List<String> ngrams = getCleanNGrams(tokens, i);
				if (ngrams != null) {
					for (String ngram : ngrams) {
						if (this.n == 1) {
							// FIXME: This further splits unigrams if 
							// the cleaning process inserts spaces into
							// them.  It's currently a hack for the
							// text classification project, but it should
							// be done differently. It shouldn't affect
							// anything if the clean function never inserts
							// spaces
							String[] ngramParts = ngram.split("\\s+");
							for (String ngramPart : ngramParts) {
								if (!retNgrams.containsKey(ngramPart))
									retNgrams.put(ngramPart, 1);
								else
									retNgrams.put(ngramPart, retNgrams.get(ngramPart) + 1);
							}
						} else {
							if (!retNgrams.containsKey(ngram))
								retNgrams.put(ngram, 1);
							else
								retNgrams.put(ngram, retNgrams.get(ngram) + 1);
						}
					}
				}
			}
		}
		return retNgrams;
	}

	@Override
	public String getGenericName() {
		return "NGramSentence";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureNGramSentence<D, L>();
	} 

}
