package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.constraint.Constraint;

public class SupervisedModelPartition<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private L defaultLabel;
	private String[] hyperParameterNames = { "defaultLabel" };
	private List<Constraint<D, L>> constraints;
	private List<SupervisedModel<D, L>> models;
	private List<List<Feature<D, L>>> features;
	
	public SupervisedModelPartition() {
		this.constraints = new ArrayList<Constraint<D, L>>();
		this.models = new ArrayList<SupervisedModel<D, L>>();
		this.features = new ArrayList<List<Feature<D, L>>>();
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data) {
		for (int i = 0; i < this.models.size(); i++) {
			FeaturizedDataSet<D, L> modelData = this.constraints.get(i).getSatisfyingSubset(data, this.labelMapping);
			for (Feature<D, L> feature : this.features.get(i)) {
				if (!feature.init(modelData) || !modelData.addFeature(feature))
					return false;
			}
			
			if (!this.models.get(i).train(modelData))
				return false;
		}
		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posterior = new HashMap<D, Map<L, Double>>();
		for (int i = 0; i < this.models.size(); i++) {
			FeaturizedDataSet<D, L> modelData = this.constraints.get(i).getSatisfyingSubset(data, this.labelMapping);
			for (Feature<D, L> feature : this.features.get(i))
				if (!feature.init(modelData) || !modelData.addFeature(feature))
					return null;
			
			Map<D, Map<L, Double>> modelPosterior = this.models.get(i).posterior(modelData);
			for (Entry<D, Map<L, Double>> entry : modelPosterior.entrySet()) {
				if (posterior.containsKey(entry.getKey()))
					continue;
				Map<L, Double> p = new HashMap<L, Double>();
				for (L validLabel : this.validLabels) {
					if (!entry.getValue().containsKey(validLabel)) {
						p.put(validLabel, 0.0);
					} else {
						p.put(validLabel, entry.getValue().get(validLabel));
					}
					modelPosterior.put(entry.getKey(), p);
				}
			}
		}
		return posterior;
	}

	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		// FIXME
		return false;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public String getGenericName() {
		return "Partition";
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		if (parameter.equals("defaultLabel"))
			return this.defaultLabel.toString();
		else
			return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("defaultLabel"))
			this.defaultLabel = datumTools.labelFromString(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	protected String[] getHyperParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelPartition<D, L>();
	}

}
