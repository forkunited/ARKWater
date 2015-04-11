package ark.data.feature.fn;

import ark.parse.ARKParsableFunction;
import ark.data.Context;

public abstract class Fn<S, T> extends ARKParsableFunction {
	/**
	 * @param input
	 * @return output
	 */
	public abstract T compute(S input);
	
	/**
	 * @param context
	 * @return a generic instance of the function.  This is used when deserializing
	 * the parameters for the function from a configuration file
	 */
	public abstract Fn<S, T> makeInstance(Context<?, ?> context);
}
