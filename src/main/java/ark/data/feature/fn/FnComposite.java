package ark.data.feature.fn;

import java.util.List;

import ark.data.Context;
import ark.data.annotation.nlp.TokenSpan;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public abstract class FnComposite<S, T, U> extends Fn<S, T> {
	public static class FnCompositeTokenSpan extends FnComposite<List<TokenSpan>, List<TokenSpan>, List<TokenSpan>> {
		public FnCompositeTokenSpan() {
			super();
		}
		
		public FnCompositeTokenSpan(Context<?, ?> context) {
			super(context);
		}

		@Override
		protected Fn<List<TokenSpan>, List<TokenSpan>> constructParameterF(Obj parameterValue) {
			return this.context.getMatchOrConstructTokenSpanFn(parameterValue);
		}

		@Override
		protected Fn<List<TokenSpan>, List<TokenSpan>> constructParameterG(Obj parameterValue) {
			return this.context.getMatchOrConstructTokenSpanFn(parameterValue);
		}

		@Override
		public Fn<List<TokenSpan>, List<TokenSpan>> makeInstance(
				Context<?, ?> context) {
			return new FnCompositeTokenSpan(context);
		}
	}
	
	public static class FnCompositeStr extends FnComposite<List<String>, List<String>, List<String>> {
		public FnCompositeStr() {
			super();
		}
		
		public FnCompositeStr(Context<?, ?> context) {
			super(context);
		}

		@Override
		protected Fn<List<String>, List<String>> constructParameterF(Obj parameterValue) {
			return this.context.getMatchOrConstructStrFn(parameterValue);
		}

		@Override
		protected Fn<List<String>, List<String>> constructParameterG(Obj parameterValue) {
			return this.context.getMatchOrConstructStrFn(parameterValue);
		}

		@Override
		public Fn<List<String>, List<String>> makeInstance(
				Context<?, ?> context) {
			return new FnCompositeStr(context);
		}
	}
	
	public static class FnCompositeTokenSpanTokenSpanStr extends FnComposite<List<TokenSpan>, List<String>, List<TokenSpan>> {
		public FnCompositeTokenSpanTokenSpanStr() {
			super();
		}
		
		public FnCompositeTokenSpanTokenSpanStr(Context<?, ?> context) {
			super(context);
		}

		@Override
		protected Fn<List<TokenSpan>, List<String>> constructParameterF(Obj parameterValue) {
			return this.context.getMatchOrConstructTokenSpanStrFn(parameterValue);
		}

		@Override
		protected Fn<List<TokenSpan>, List<TokenSpan>> constructParameterG(Obj parameterValue) {
			return this.context.getMatchOrConstructTokenSpanFn(parameterValue);
		}

		@Override
		public Fn<List<TokenSpan>, List<String>> makeInstance(
				Context<?, ?> context) {
			return new FnCompositeTokenSpanTokenSpanStr(context);
		}
	}
	
	public static class FnCompositeTokenSpanStrStr extends FnComposite<List<TokenSpan>, List<String>, List<String>> {
		public FnCompositeTokenSpanStrStr() {
			super();
		}
		
		public FnCompositeTokenSpanStrStr(Context<?, ?> context) {
			super(context);
		}

		@Override
		protected Fn<List<String>, List<String>> constructParameterF(Obj parameterValue) {
			return this.context.getMatchOrConstructStrFn(parameterValue);
		}

		@Override
		protected Fn<List<TokenSpan>, List<String>> constructParameterG(Obj parameterValue) {
			return this.context.getMatchOrConstructTokenSpanStrFn(parameterValue);
		}

		@Override
		public Fn<List<TokenSpan>, List<String>> makeInstance(
				Context<?, ?> context) {
			return new FnCompositeTokenSpanStrStr(context);
		}
	}
	
	private String[] parameterNames = { "f", "g" };
	protected Fn<U, T> f;
	protected Fn<S, U> g;
	protected Context<?, ?> context;
	
	protected abstract Fn<U, T> constructParameterF(Obj parameterValue);
	protected abstract Fn<S, U> constructParameterG(Obj parameterValue);

	public FnComposite() {
		
	}
	
	public FnComposite(Context<?, ?> context) {
		this.context = context;
	}
	
	@Override
	public T compute(S input) {
		return this.f.compute(this.g.compute(input));
	}

	@Override
	public String getGenericName() {
		return "Composite";
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("f"))
			return this.f.toParse();
		else if (parameter.equals("g"))
			return this.g.toParse();
		else 
			return null;
	}
	
	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("f")) {
			this.f = constructParameterF(parameterValue);
			if (this.f == null)
				return false;
		} else if (parameter.equals("g")) {
			this.g = constructParameterG(parameterValue);
			if (this.g == null)
				return false;
		} else 
			return false;
		
		return true;
	}
	
	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {		
		return true;
	}

	@Override
	protected AssignmentList toParseInternal() {
		return null;
	}
}
