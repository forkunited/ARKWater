package ark.data.feature;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.cluster.Clusterer;
import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

public class FeatureGramCluster<D extends Datum<L>, L> extends FeatureGram<D, L> {	
	protected Clusterer<TokenSpan> clusterer;
	
	public FeatureGramCluster() {
		super();
		
		this.clusterer = null;
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "clusterer";
	}


	@Override
	public String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("clusterer"))
			return (this.clusterer == null) ? "None" : this.clusterer.getName();
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("clusterer"))
			this.clusterer = datumTools.getDataTools().getTokenSpanClusterer(parameterValue);
		else
			return false;
		
		return true;
	}
	
	@Override
	protected Map<String, Integer> getGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Map<String, Integer> retGrams = new HashMap<String, Integer>();
		
		for (TokenSpan tokenSpan : tokenSpans) {			
			List<String> clusters = this.clusterer.getClusters(tokenSpan);
			
			for (String cluster : clusters) {
				if (!retGrams.containsKey(cluster))
					retGrams.put(cluster, 1);
				else
					retGrams.put(cluster, retGrams.get(cluster) + 1);
			}
		}
			
		return retGrams;
	}
	
	@Override
	public String getGenericName() {
		return "GramCluster";
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureGramCluster<D, L>();
	}
}
