package ark.model;

import java.io.BufferedReader;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;

public class SupervisedModelLabelDistribution<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private Map<L, Double> labelDistribution;
	
	public SupervisedModelLabelDistribution() {
		this.labelDistribution = new HashMap<L, Double>();
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		double total = 0;
		
		for (L label : this.validLabels)
			this.labelDistribution.put(label, 0.0);
		
		for (D datum : data) {
			L label = mapValidLabel(datum.getLabel());
			if (label == null)
				continue;
			
			this.labelDistribution.put(label, this.labelDistribution.get(label) + 1.0);
			total += 1.0;
		}
		
		for (Entry<L, Double> entry : this.labelDistribution.entrySet()) {
			entry.setValue(entry.getValue() / total);
		}

		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posterior = new HashMap<D, Map<L, Double>>();
		for (D datum : data) {
			posterior.put(datum, this.labelDistribution);
		}
		return posterior;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) {
		// TODO: Serialize the distribution.  This isn't necessary for now because we never save
		// this kind of model since it's just used to compute the majority baseline
		return true;
	}

	@Override
	protected boolean serializeParameters(Writer writer) {
		// TODO: Serialize the distribution.  This isn't necessary for now because we never save
		// this kind of model since it's just used to compute the majority baseline
		return true;
	}
	
	@Override
	protected String[] getHyperParameterNames() {
		return new String[0];
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelLabelDistribution<D, L>();
	}

	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) {
		return true;
	}

	@Override
	public String getGenericName() {
		return "LabelDistribution";
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		return true;
	}
}
