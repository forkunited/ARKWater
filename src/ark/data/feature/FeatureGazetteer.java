package ark.data.feature;

import java.util.HashMap;
import java.util.Map;

import ark.data.Gazetteer;
import ark.data.annotation.Datum;

public abstract class FeatureGazetteer<D extends Datum<L>, L> extends Feature<D, L> {
	protected enum ExtremumType {
		Minimum,
		Maximum
	}
	
	protected FeatureGazetteer.ExtremumType extremumType;
	
	protected Gazetteer gazetteer;
	protected Datum.Tools.StringExtractor<D, L> stringExtractor;
	protected String[] parameterNames = {"gazetteer", "stringExtractor"};
	
	protected abstract double computeExtremum(String str);
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>(1);
		vector.put(0, computeExtremum(datum));
		return vector;
	}

	
	protected double computeExtremum(D datum) {
		String[] strs = this.stringExtractor.extract(datum);
		double extremum = (this.extremumType == FeatureGazetteer.ExtremumType.Maximum) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		for (String str : strs) {
			double curExtremum = computeExtremum(str);
			if ((this.extremumType == FeatureGazetteer.ExtremumType.Maximum && curExtremum > extremum)
					|| (this.extremumType == FeatureGazetteer.ExtremumType.Minimum && curExtremum < extremum))
				extremum = curExtremum;	
		}
		return extremum;
	}
	
	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("gazetteer"))
			return (this.gazetteer == null) ? null : this.gazetteer.getName();
		else if (parameter.equals("stringExtractor"))
			return (this.stringExtractor == null) ? null : this.stringExtractor.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("gazetteer"))
			this.gazetteer = datumTools.getDataTools().getGazetteer(parameterValue);
		else if (parameter.equals("stringExtractor"))
			this.stringExtractor = datumTools.getStringExtractor(parameterValue);
		else 
			return false;
		return true;
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		return true;
	}
	
	@Override
	protected String getVocabularyTerm(int index) {
		return null;
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		return true;
	}

	@Override
	public int getVocabularySize() {
		return 1;
	}
}
