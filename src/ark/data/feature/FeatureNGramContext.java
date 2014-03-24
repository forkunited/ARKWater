package ark.data.feature;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

public class FeatureNGramContext<D extends Datum<L>, L> extends FeatureNGram<D, L> {
	private int contextWindowSize;
	
	public FeatureNGramContext() {
		super();
		
		this.contextWindowSize = 0;
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "contextWindowSize";
	}
	
	@Override
	protected Set<String> getNGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		HashSet<String> retNgrams = new HashSet<String>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			List<String> tokens = tokenSpan.getDocument().getSentenceTokens(tokenSpan.getSentenceIndex());
			if (tokens == null)
				continue;
			int startIndex = Math.max(0, tokenSpan.getStartTokenIndex() - this.contextWindowSize);
			int endIndex = Math.min(tokens.size(), tokenSpan.getEndTokenIndex() + this.contextWindowSize) - this.n + 1;
			for (int i = startIndex; i < endIndex; i++) {				
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
