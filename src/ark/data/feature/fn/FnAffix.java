package ark.data.feature.fn;

import java.util.ArrayList;
import java.util.List;

import ark.data.Context;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnAffix extends Fn<List<String>, List<String>> {
	public enum Type {
		SUFFIX,
		PREFIX
	}
	
	private Type type = Type.SUFFIX;
	private int n = 3;
	private String[] parameterNames = { "type", "n" };
	private Context<?, ?> context;
	
	public FnAffix() {
		
	}
	
	public FnAffix(Context<?, ?> context) {
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
		else if (parameter.equals("type"))
			return Obj.stringValue(this.type.toString());
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("n"))
			this.n = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("type"))
			this.type = Type.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	public List<String> compute(List<String> input) {
		List<String> affixes = new ArrayList<String>(input.size());
		
		for (String str : input) {
			if (str.length() <= this.n)
				continue;
				
			String affix = (this.type == Type.SUFFIX) ? 
					str.substring(str.length() - this.n, str.length()) 
					: str.substring(0, this.n);
			
			affixes.add(affix);
		}
		
		return affixes;
	}

	@Override
	public Fn<List<String>, List<String>> makeInstance(Context<?, ?> context) {
		return new FnAffix();
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
		return "Affix";
	}

}
