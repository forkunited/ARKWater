package ark.data.feature.fn;

import java.util.ArrayList;
import java.util.List;

import ark.data.Context;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnFilter extends Fn<List<String>, List<String>> {
	public enum Type {
		SUFFIX,
		PREFIX
	}
	
	private String[] parameterNames = { "filter", "type" };
	private String filter = "";
	private Type type = Type.SUFFIX;
	
	private Context<?, ?> context;

	public FnFilter() {
		
	}
	
	public FnFilter(Context<?, ?> context) {
		this.context = context;
	}
	
	
	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("type"))
			return Obj.stringValue(this.type.toString());
		else if (parameter.equals("filter"))
			return Obj.stringValue(this.filter);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("type"))
			this.type = Type.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("filter"))
			this.filter = this.context.getMatchValue(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	public List<String> compute(List<String> input) {
		if (this.filter.length() == 0)
			return input;
		
		List<String> filtered = new ArrayList<String>();
		for (String str : input) {
			if (matchesFilter(str))
				filtered.add(str);
		}
		
		return filtered;
	}
	
	private boolean matchesFilter(String str) {
		if (this.type == Type.SUFFIX)
			return str.endsWith(this.filter);
		else
			return str.startsWith(this.filter);
	}

	@Override
	public Fn<List<String>, List<String>> makeInstance(Context<?, ?> context) {
		return new FnFilter(context);
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
		return "Filter";
	}

}
