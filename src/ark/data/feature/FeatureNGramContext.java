package ark.data.feature;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

/**
 * For each datum d FeatureNGramContext computes a
 * vector:
 * 
 * <c(v_1\in C(T(d),k)), c(v_2 \in C(T(d),k)), ... , c(v_n \in C(T(d),k))>
 * 
 * Where T is a token extractor, C(T(d), k) computes a context of window-size
 * k of n-grams surrounding the tokens given by T(d) in a source text document,
 * and c(v \in S) computes the number of occurrences of n-gram v in S.  The resulting
 * vector is given to methods in ark.data.feature.FeatureNGram to be normalized
 * and scaled in some way.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class FeatureNGramContext<D extends Datum<L>, L> extends FeatureNGram<D, L> {
	private int contextWindowSize;
	
	public FeatureNGramContext() {
		super();
		
		this.contextWindowSize = 0;
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "contextWindowSize";
	}
	
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
			int startIndex = Math.max(0, tokenSpan.getStartTokenIndex() - this.contextWindowSize);
			int endIndex = Math.min(tokens.size(), tokenSpan.getEndTokenIndex() + this.contextWindowSize) - this.n + 1;
			for (int i = startIndex; i < endIndex; i++) {				
				List<String> ngrams = getCleanNGrams(tokens, i);
				if (ngrams != null) {
					for (String ngram : ngrams) {
						if (!retNgrams.containsKey(ngram))
							retNgrams.put(ngram, 1);
						else
							retNgrams.put(ngram, retNgrams.get(ngram) + 1);
					}
				}
			}
		}
		return retNgrams;
	}

	@Override
	public String getGenericName() {
		return "NGramContext";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureNGramContext<D, L>();
	}
	
	@Override
	protected String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("contextWindowSize"))
			return String.valueOf(this.contextWindowSize);
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("contextWindowSize"))
			this.contextWindowSize = Integer.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}
}
