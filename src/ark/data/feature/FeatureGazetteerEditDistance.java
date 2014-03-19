package ark.data.feature;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.StringUtil;

public class FeatureGazetteerEditDistance<D extends Datum<L>, L> extends FeatureGazetteer<D, L> {
	private DataTools.StringPairMeasure editDistanceMeasure;
	
	public FeatureGazetteerEditDistance() {
		this.extremumType = FeatureGazetteer.ExtremumType.Minimum;
		
		this.editDistanceMeasure = new DataTools.StringPairMeasure() {
			public double compute(String str1, String str2) {
				return StringUtil.levenshteinDistance(str1, str2)/((double)(str1.length()+str2.length()));
			}
		};
	}
	
	@Override
	protected double computeExtremum(String str) {
		return this.gazetteer.min(str, this.editDistanceMeasure);
	}

	@Override
	public String getGenericName() {
		return "GazetteerEditDistance";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureGazetteerEditDistance<D, L>();
	}
}
