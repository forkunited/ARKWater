package ark.data.feature;

import java.util.Arrays;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.StringUtil;

public class FeatureGazetteerPrefixTokens<D extends Datum<L>, L> extends FeatureGazetteer<D, L> {
	private DataTools.StringPairMeasure prefixTokensMeasure;
	private int minTokens;
	
	public FeatureGazetteerPrefixTokens() {
		this.extremumType = FeatureGazetteer.ExtremumType.Maximum;
		
		this.prefixTokensMeasure = new DataTools.StringPairMeasure() {
			public double compute(String str1, String str2) {
				return StringUtil.prefixTokenOverlap(str1, str2);
			}
		};
		
		this.minTokens = 2;
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "minTokens";
	}
	
	@Override
	protected double computeExtremum(String str) {
		double tokenPrefixCount = this.gazetteer.max(str, this.prefixTokensMeasure);
		
		if (tokenPrefixCount >= this.minTokens)
			return 1.0;
		else
			return 0.0;
	}

	@Override
	public String getGenericName() {
		return "GazetteerPrefixTokens";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureGazetteerPrefixTokens<D, L>();
	}

	@Override
	protected String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("minTokens"))
			return String.valueOf(this.minTokens);
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, DataTools dataTools, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, dataTools, datumTools))
			return true;
		else if (parameter.equals("minTokens"))
			this.minTokens = Integer.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}
}
