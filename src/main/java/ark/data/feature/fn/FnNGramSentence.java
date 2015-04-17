package ark.data.feature.fn;

import java.util.Arrays;
import java.util.List;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.Obj;

public class FnNGramSentence extends FnNGram {
	private boolean noSpan = false;
	
	public FnNGramSentence() {
		this(null);
	}
	
	public FnNGramSentence(Context<?, ?> context) {
		super(context);
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "noSpan";
	}
	
	@Override
	protected boolean getNGrams(TokenSpan tokenSpan, List<TokenSpan> ngrams) {
		int s = tokenSpan.getSentenceIndex();
		int tokenCount = tokenSpan.getDocument().getSentenceTokenCount(s);
		
		for (int i = 0; i < tokenCount - this.n + 1; i++) {
			TokenSpan ngram = new TokenSpan(tokenSpan.getDocument(), s, i, i + this.n);
			
			if (!this.noSpan 
					|| 
						(!tokenSpan.containsToken(s, i) 
						&& !tokenSpan.containsToken(s, i + this.n - 1)
						&& !ngram.containsToken(s, tokenSpan.getStartTokenIndex())
						&& !ngram.containsToken(s, tokenSpan.getEndTokenIndex() - 1)
							))
				ngrams.add(ngram);
			
		}
		
		return true;
	}

	@Override
	public Fn<List<TokenSpan>, List<TokenSpan>> makeInstance(
			Context<?, ?> context) {
		return new FnNGramSentence(context);
	}
	
	@Override
	public String getGenericName() {
		return "NGramSentence";
	}
	
	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("noSpan"))
			return Obj.stringValue(String.valueOf(this.noSpan));
		else 
			return super.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("noSpan"))
			this.noSpan = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else
			return super.setParameterValue(parameter, parameterValue);
		return true;
	}
}
