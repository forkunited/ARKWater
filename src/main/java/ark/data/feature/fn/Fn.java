package ark.data.feature.fn;

import java.util.HashMap;
import java.util.Map;

import ark.parse.ARKParsableFunction;
import ark.data.Context;

public abstract class Fn<S, T> extends ARKParsableFunction {
	public enum CacheMode {
		MANY,
		ONE,
		NONE
	}
	
	private Map<String, T> manyCache;
	private T oneCache;
	
	protected void addToManyCache(String id, T output) {
		this.manyCache.put(id, output);
	}
	
	protected T lookupManyCache(String id) {
		return this.manyCache.get(id);
	}
	
	protected void initializeManyCache() {
		if (this.manyCache == null)
			this.manyCache = new HashMap<String, T>();
	}
	
	protected void addToOneCache(T output) {
		this.oneCache = output;
	}
	
	protected T lookupOneCache() {
		return this.oneCache;
	}
	
	protected void initializeOneCache() {

	}
	
	public T manyCachedCompute(S input, String id) {
		initializeManyCache();
		
		T output = lookupManyCache(id);
		if (output == null)
			output = compute(input);
		
		addToManyCache(id, output);
		
		return output;
	}
	
	public T oneCachedCompute(S input) {
		initializeOneCache();
		
		T output = lookupOneCache();
		if (output != null)
			return output;
		
		output = compute(input);
		
		addToOneCache(output);
		
		return output;
	}
	
	public void clearCaches() {
		this.manyCache = null;
		this.oneCache = null;
	}
	
	public T compute(S input, String id, CacheMode cacheMode) {
		if (cacheMode == CacheMode.MANY)
			return manyCachedCompute(input, id);
		else if (cacheMode == CacheMode.ONE)
			return oneCachedCompute(input);
		else
			return compute(input);
	}
	
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
