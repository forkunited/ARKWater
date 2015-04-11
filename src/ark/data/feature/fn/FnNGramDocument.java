package ark.data.feature.fn;

import java.util.Arrays;
import java.util.List;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.Obj;

public class FnNGramDocument extends FnNGram {
	private boolean noSentence = false;
	
	public FnNGramDocument() {
		this(null);
	}
	
	public FnNGramDocument(Context<?, ?> context) {
		super(context);
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "noSentence";
	}
	
	@Override
	protected boolean getNGrams(TokenSpan tokenSpan, List<TokenSpan> ngrams) {
		int sentenceCount = tokenSpan.getDocument().getSentenceCount();
		
		for (int i = 0; i < sentenceCount; i++) {
			if (!this.noSentence || i != tokenSpan.getSentenceIndex()) {
				int tokenCount = tokenSpan.getDocument().getSentenceTokenCount(i);
				for (int j = 0; j < tokenCount; j++) {
					ngrams.add(new TokenSpan(tokenSpan.getDocument(), i, j, j + this.n));
				}
			}
		}
		
		return true;
	}

	@Override
	public Fn<List<TokenSpan>, List<TokenSpan>> makeInstance(
			Context<?, ?> context) {
		return new FnNGramDocument();
	}
	
	@Override
	public String getGenericName() {
		return "NGramDocument";
	}
	
	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("noSentence"))
			return Obj.stringValue(String.valueOf(this.noSentence));
		else 
			return super.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("noSentence"))
			this.noSentence = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else
			return super.setParameterValue(parameter, parameterValue);
		return true;
	}
}
