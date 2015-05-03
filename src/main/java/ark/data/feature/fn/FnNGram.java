package ark.data.feature.fn;

import java.util.Collection;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public abstract class FnNGram extends Fn<TokenSpan, TokenSpan> {
	protected String[] parameterNames = { "n" };
	protected int n = 1;
	protected Context<?, ?> context;

	public FnNGram() {
		
	}
	
	public FnNGram(Context<?, ?> context) {
		this.context = context;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("n"))
			return Obj.stringValue(String.valueOf(this.n));
		else 
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("n"))
			this.n = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	public <C extends Collection<TokenSpan>> C compute(Collection<TokenSpan> input, C output) {
		for (TokenSpan tokenSpan : input)
			getNGrams(tokenSpan, output);
		
		return output;
	}
	
	protected abstract boolean getNGrams(TokenSpan tokenSpan, Collection<TokenSpan> ngrams);
	

	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {
		return true;
	}

	@Override
	protected AssignmentList toParseInternal() {
		return null;
	}
}
