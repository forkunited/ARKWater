package ark.data.feature;

import java.util.Arrays;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.StringUtil;

/**
 * For datum d, string extractor S, and gazetteer G, 
 * FeatureGazetteerInitialism computes
 * 
 * max_{g\in G} 1(S(d) is an initialism for g)
 * 
 * An 'allowPrefix' parameter determines whether S(d) must
 * be a full initialism, or just a partial (prefix) initialism.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 *
 */
public class FeatureGazetteerInitialism<D extends Datum<L>, L> extends FeatureGazetteer<D, L> {
	private DataTools.StringPairMeasure initialismMeasure;
	private boolean allowPrefix;
	
	public FeatureGazetteerInitialism() {
		this.extremumType = FeatureGazetteer.ExtremumType.Maximum;
		
		this.initialismMeasure = new DataTools.StringPairMeasure() {
			public double compute(String str1, String str2) {
				if (StringUtil.isInitialism(str1, str2, allowPrefix))
					return 1.0;
				else
					return 0.0;
			}
		};
		
		this.allowPrefix = false;
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "allowPrefix";
	}
	
	@Override
	protected double computeExtremum(String str) {
		return this.gazetteer.max(str, this.initialismMeasure);
	}

	@Override
	public String getGenericName() {
		return "GazetteerInitialism";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureGazetteerInitialism<D, L>();
	}
	
	@Override
	protected String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("allowPrefix"))
			return String.valueOf(this.allowPrefix);
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("allowPrefix"))
			this.allowPrefix = Boolean.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}
}
