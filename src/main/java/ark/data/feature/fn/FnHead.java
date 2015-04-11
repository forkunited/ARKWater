package ark.data.feature.fn;

import java.util.ArrayList;
import java.util.List;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnHead extends Fn<List<TokenSpan>, List<TokenSpan>> {
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
	public List<TokenSpan> compute(List<TokenSpan> input) {
		List<TokenSpan> heads = new ArrayList<TokenSpan>(input.size());
		
		for (TokenSpan span : input) {
			if (span.getLength() > 0)
				heads.add(span.getSubspan(span.getLength() - 1, span.getLength()));
		}
		
		return heads;
	}

	@Override
	public Fn<List<TokenSpan>, List<TokenSpan>> makeInstance(
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
