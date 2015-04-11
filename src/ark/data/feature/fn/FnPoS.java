package ark.data.feature.fn;

import java.util.ArrayList;
import java.util.List;

import ark.cluster.ClustererTokenSpanPoSTag;
import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnPoS extends Fn<List<TokenSpan>, List<String>> {
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
	public List<String> compute(List<TokenSpan> input) {
		List<String> pos = new ArrayList<String>(input.size());
		
		for (TokenSpan tokenSpan : input) {
			pos.addAll(this.clusterer.getClusters(tokenSpan));
		}
		
		return pos;
	}

	@Override
	public Fn<List<TokenSpan>, List<String>> makeInstance(Context<?, ?> context) {
		return new FnPoS();
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
