package ark.data.feature.fn;

import java.util.Collection;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnHead extends Fn<TokenSpan, TokenSpan> {
	private String[] parameterNames = {};
	
	public FnHead() {
		
	}
	
	public FnHead(Context<?, ?> context) {
		
	}
	
	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		return false;
	}

	@Override
	public <C extends Collection<TokenSpan>> C compute(Collection<TokenSpan> input, C output) {
		for (TokenSpan span : input) {
			if (span.getLength() > 0)
				output.add(span.getSubspan(span.getLength() - 1, span.getLength()));
		}
		
		return output;
	}

	@Override
	public Fn<TokenSpan, TokenSpan> makeInstance(
			Context<?, ?> context) {
		return new FnHead(context);
	}

	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {
		return true;
	}

	@Override
	protected AssignmentList toParseInternal() {
		return null;
	}

	@Override
	public String getGenericName() {
		return "Head";
	}

}
