package ark.parse;

import java.util.Map;
import java.util.Map.Entry;

public abstract class ARKParsableFunction extends ARKParsable implements Parameterizable {	
	@Override
	public Obj toParse() {
		return toParse(true);
	}
	
	public Obj toParse(boolean includeInternal) {
		String[] parameters = getParameterNames();
		AssignmentList parameterList = new AssignmentList();
		for (String parameterName : parameters)
			parameterList.add(Assignment.assignmentUntyped(parameterName, getParameterValue(parameterName)));
		
		return Obj.function(getGenericName(), parameterList, (includeInternal) ? toParseInternal() : null);
	}

	@Override
	protected boolean fromParseHelper(Obj obj) {
		Obj.Function function = (Obj.Function)obj;
		
		AssignmentList parameterList = function.getParameters();
		if (parameterList.size() > 0) {
			if (parameterList.get(0).getName() != null) {
				for (int i = 0; i < parameterList.size(); i++) {
					if (!setParameterValue(parameterList.get(i).getName(), parameterList.get(i).getValue()))
						return false;
				}
			} else {
				String[] parameters = getParameterNames();
				for (int i = 0; i < parameterList.size(); i++)
					if (!setParameterValue(parameters[i], parameterList.get(i).getValue()))
						return false;
			}
		}
		
		return fromParseInternal(function.getInternalAssignments());
	}
	
	public boolean setParameterValues(Map<String, Obj> parameterValues) {
		for (Entry<String, Obj> entry : parameterValues.entrySet())
			if (!setParameterValue(entry.getKey(), entry.getValue()))
				return false;
		
		return true;
	}
	
	protected abstract boolean fromParseInternal(AssignmentList internalAssignments);
	protected abstract AssignmentList toParseInternal();
	public abstract String getGenericName();
}