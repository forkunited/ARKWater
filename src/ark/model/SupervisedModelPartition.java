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
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.SerializationUtil;

public class SupervisedModelPartition<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private L defaultLabel;
	private String[] hyperParameterNames = { "defaultLabel" };
	private List<String> orderedModels;
	private Map<String, Constraint<D, L>> constraints;
	private Map<String, SupervisedModel<D, L>> models;
	private Map<String, List<Feature<D, L>>> features;
	
	public SupervisedModelPartition() {
		this.orderedModels = new ArrayList<String>();
		this.constraints = new HashMap<String, Constraint<D, L>>();
		this.models = new HashMap<String, SupervisedModel<D, L>>();
		this.features = new HashMap<String, List<Feature<D, L>>>();
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		Map<D, L> fixedLabels = new HashMap<D, L>();
		for (int i = 0; i < this.orderedModels.size(); i++) {
			String modelName = this.orderedModels.get(i);
			FeaturizedDataSet<D, L> modelData = this.constraints.get(modelName).getSatisfyingSubset(data, this.labelMapping);
			FeaturizedDataSet<D, L> modelTestData = this.constraints.get(modelName).getSatisfyingSubset(testData, this.labelMapping);
		
			for (Feature<D, L> feature : this.features.get(modelName)) {
				if (!feature.init(modelData) || !modelData.addFeature(feature))
					return false;
			}
			
			if (!this.models.get(modelName).train(modelData, modelTestData, evaluations))
				return false;
			
			fixedLabels.putAll(this.models.get(modelName).classify(modelData));
			for (int j = i + 1; j < this.orderedModels.size(); j++)
				this.models.get(this.orderedModels.get(j)).fixDatumLabels(fixedLabels);
		}
		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posterior = new HashMap<D, Map<L, Double>>();
		Map<D, L> fixedLabels = new HashMap<D, L>();
		for (int i = 0; i < this.orderedModels.size(); i++) {
			String modelName = this.orderedModels.get(i);
			FeaturizedDataSet<D, L> modelData = this.constraints.get(modelName).getSatisfyingSubset(data, this.labelMapping);
			for (Feature<D, L> feature : this.features.get(modelName))
				if (!modelData.addFeature(feature))
					return null;
			
			Map<D, Map<L, Double>> modelPosterior = this.models.get(modelName).posterior(modelData);
			for (Entry<D, Map<L, Double>> pEntry : modelPosterior.entrySet()) {
				if (posterior.containsKey(pEntry.getKey()))
					continue;
				Map<L, Double> p = new HashMap<L, Double>();
				L bestLabel = this.defaultLabel;
				double bestLabelValue = 0.0;
				for (L validLabel : this.validLabels) {
					if (!pEntry.getValue().containsKey(validLabel)) {
						p.put(validLabel, 0.0);
					} else {
						double labelValue = pEntry.getValue().get(validLabel);
						p.put(validLabel, labelValue);
						if (labelValue >= bestLabelValue) {
							bestLabel = validLabel;
							bestLabelValue = labelValue;
						}
					}
				}
				posterior.put(pEntry.getKey(), p);
				fixedLabels.put(pEntry.getKey(), bestLabel);
			}
			
			for (int j = i + 1; j < this.orderedModels.size(); j++)
				this.models.get(this.orderedModels.get(j)).fixDatumLabels(fixedLabels);
		}
		
		for (D datum : data) { // Mark remaining data with default label
			Map<L, Double> p = new HashMap<L, Double>();
			for (L validLabel : this.validLabels)
				p.put(validLabel, 0.0);
			p.put(this.defaultLabel, 1.0);
			
			if (!posterior.containsKey(datum))
				posterior.put(datum, p);
		}
		
		return posterior;
	}

	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		String[] nameParts = name.split("_");
		String type = nameParts[0];
		String modelReference = nameParts[1];
		
		if (type.equals("model")) {
			String modelName = SerializationUtil.deserializeGenericName(reader);
			SupervisedModel<D, L> model = datumTools.makeModelInstance(modelName);
			if (!model.deserialize(reader, false, false, datumTools, modelReference))
				return false;
			this.orderedModels.add(modelReference);
			this.models.put(modelReference, model);
		} else if (type.equals("feature")) {
			String featureReference = null;
			boolean ignored = false;
			if (nameParts.length > 2)
				featureReference = nameParts[2];
			if (nameParts.length > 3)
				ignored = true;
			
			String featureName = SerializationUtil.deserializeGenericName(reader);
			Feature<D, L> feature = datumTools.makeFeatureInstance(featureName);
			if (!feature.deserialize(reader, false, false, datumTools, featureReference, ignored))
				return false;
			
			if (!this.features.containsKey(modelReference))
				this.features.put(modelReference, new ArrayList<Feature<D, L>>());
			this.features.get(modelReference).add(feature);
		} else if (type.equals("constraint")) {
			this.constraints.put(modelReference, Constraint.<D,L>fromString(reader.readLine()));
		}
		
		return true;
	}

	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		// FIXME Do this later when necessary
		return true;
	}
	
	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		for (String modelName : this.orderedModels) {
			writer.write("\tmodel" + "_" + modelName + "=" + this.models.get(modelName).toString(false) + "\n");
			// FIXME Include extra info in model output
			
			writer.write("\tconstraint_" + modelName + "=" + this.constraints.get(modelName).toString() + "\n");
			List<Feature<D, L>> features = this.features.get(modelName);
			for (int i = 0; i < features.size(); i++) {
				writer.write("\tfeature_" + modelName);
				if (features.get(i).getReferenceName() != null)
					writer.write("_" + features.get(i).getReferenceName());
				if (features.get(i).isIgnored())
					writer.write("_ignored");
				writer.write("=" + features.get(i).toString(false) + "\n");
			}
		}
		
		return true;
	}

	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		for (Entry<String, SupervisedModel<D, L>> entry : this.models.entrySet()) {
			writer.write("BEGIN PARAMETERS " + entry.getKey() + "\n\n");
			entry.getValue().serializeParameters(writer);
			writer.write("\nEND PARAMETERS " +  entry.getKey() + "\n\n");
			
			// Write features (that have been initialized)
			writer.write("BEGIN FEATURES " +  entry.getKey() + "\n\n");
			for (int j = 0; j < this.features.get(entry.getKey()).size(); j++) {
				this.features.get(entry.getKey()).get(j).serialize(writer);
				writer.write("\n\n");
			}
			writer.write("END FEATURES " + entry.getKey() + "\n\n");
		}
		
		return true;
	}

	@Override
	public String getGenericName() {
		return "Partition";
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		int firstUnderscoreIndex = parameter.indexOf("_");
		if (parameter.equals("defaultLabel"))
			return (this.defaultLabel == null) ? null : this.defaultLabel.toString();
		else if (firstUnderscoreIndex >= 0) {
			String modelReference = parameter.substring(0, firstUnderscoreIndex);
			String modelParameter = parameter.substring(firstUnderscoreIndex + 1);
			this.models.get(modelReference).getHyperParameterValue(modelParameter);
		}
		
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		int firstUnderscoreIndex = parameter.indexOf("_");
		if (parameter.equals("defaultLabel")) {
			this.defaultLabel = (parameterValue == null) ? null : datumTools.labelFromString(parameterValue);
			return true;
		} else if (firstUnderscoreIndex >= 0) {
			String modelReference = parameter.substring(0, firstUnderscoreIndex);
			String modelParameter = parameter.substring(firstUnderscoreIndex + 1);
			this.models.get(modelReference).setHyperParameterValue(modelParameter, parameterValue, datumTools);
		}
		return false;
	}

	@Override
	protected String[] getHyperParameterNames() {
		List<String> hyperParameterNames = new ArrayList<String>();
		for (String parameterName : this.hyperParameterNames) {
			hyperParameterNames.add(parameterName);
		}
		
		if (this.models != null) {
			for (SupervisedModel<D, L> model : this.models.values()) {
				String[] modelParameterNames = model.getHyperParameterNames();
				for (String parameterName : modelParameterNames) {
					hyperParameterNames.add(model.getReferenceName() + "_" + parameterName);
				}
			}
		}
		
		return this.hyperParameterNames;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelPartition<D, L>();
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelPartition<D, L> clone = (SupervisedModelPartition<D, L>)super.clone(datumTools, environment);
		
		// TODO If constraint code is improved, then cloning constraints will be necessary
		clone.constraints = this.constraints; 
		clone.features = new HashMap<String, List<Feature<D, L>>>();
		clone.models = new HashMap<String, SupervisedModel<D, L>>();
		clone.orderedModels = new ArrayList<String>();

		for (String model : this.orderedModels) {
			clone.orderedModels.add(model);
			clone.models.put(model, this.models.get(model).clone(datumTools, environment));
			clone.features.put(model, new ArrayList<Feature<D, L>>());
			for (int j = 0; j < this.features.get(model).size(); j++) {
				clone.features.get(model).add(this.features.get(model).get(j).clone(datumTools, environment));
			}
		}
		
		return clone;
	}

}
