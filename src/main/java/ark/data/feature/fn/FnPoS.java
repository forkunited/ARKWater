package ark.data.feature.fn;

import java.util.Collection;

import ark.cluster.ClustererTokenSpanPoSTag;
import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnPoS extends Fn<TokenSpan, String> {
	private ClustererTokenSpanPoSTag clusterer = new ClustererTokenSpanPoSTag();
	private String[] parameterNames = {  };
	
	public FnPoS() {
		
	}
	
	public FnPoS(Context<?, ?> context) {
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
	public <C extends Collection<String>> C compute(Collection<TokenSpan> input, C output) {
		for (TokenSpan tokenSpan : input) {
			output.addAll(this.clusterer.getClusters(tokenSpan));
		}
		
		return output;
	}

	@Override
	public Fn<TokenSpan, String> makeInstance(Context<?, ?> context) {
		return new FnPoS(context);
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
		return "PoS";
	}

}
