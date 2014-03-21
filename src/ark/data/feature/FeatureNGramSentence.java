package ark.data.feature;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

public class FeatureNGramSentence<D extends Datum<L>, L> extends FeatureNGram<D, L> {

	@Override
	protected Set<String> getNGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Set<String> retNgrams = new HashSet<String>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			List<String> tokens = tokenSpan.getDocument().getSentenceTokens(tokenSpan.getSentenceIndex());
			if (tokens == null)
				continue;
			for (int i = 0; i < tokens.size()-this.n+1; i++) {
				List<String> ngrams = getCleanNGrams(tokens, i);
				if (ngrams != null) {
					retNgrams.addAll(ngrams);
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
