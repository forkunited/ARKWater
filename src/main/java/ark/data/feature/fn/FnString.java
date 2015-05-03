package ark.data.feature.fn;

import java.util.Collection;

import ark.data.Context;
import ark.data.DataTools;
import ark.data.annotation.Document;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class FnString extends Fn<TokenSpan, String> {
	private DataTools.StringTransform cleanFn;
	private String[] parameterNames = { "cleanFn" };
	
	private Context<?, ?> context;
	
	public FnString() {
		
	}
	
	public FnString(Context<?, ?> context) {
		this.context = context;
	}
	
	@Override
	public String[] getParameterNames() {
		return parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("cleanFn"))
			return Obj.stringValue((this.cleanFn == null) ? "" : this.cleanFn.toString());
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("cleanFn"))
			this.cleanFn = this.context.getDatumTools().getDataTools().getCleanFn(this.context.getMatchValue(parameterValue));
		else 
			return false;
		
		return true;
	}

	@Override
	public <C extends Collection<String>> C compute(Collection<TokenSpan> input, C output) {
		for (TokenSpan tokenSpan : input) {
			StringBuilder str = new StringBuilder();
			int s = tokenSpan.getSentenceIndex();
			Document document = tokenSpan.getDocument();
			for (int i = 0; i < tokenSpan.getLength(); i++) {
				String tStr = document.getToken(s, i + tokenSpan.getStartTokenIndex());
				if (this.cleanFn != null)
					tStr = this.cleanFn.transform(tStr);
				
				str.append(tStr).append("_");
			}
			
			str.delete(str.length() - 1, str.length());
			
			output.add(str.toString());
		}
		
		return output;
	}

	@Override
	public Fn<TokenSpan, String> makeInstance(Context<?, ?> context) {
		return new FnString(context);
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
		return "String";
	}
}
