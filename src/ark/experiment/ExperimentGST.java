package ark.experiment;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.GridSearchTestValidation;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.SerializationUtil;

public class ExperimentGST<D extends Datum<L>, L> extends Experiment<D, L> {
	protected SupervisedModel<D, L> model;
	protected List<Feature<D, L>> features;
	protected Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor;
	protected Map<String, List<String>> gridSearchParameterValues;
	protected List<SupervisedModelEvaluation<D, L>> evaluations;
	protected DataSet<D, L> trainData;
	protected DataSet<D, L> devData;
	protected DataSet<D, L> testData;
	protected boolean trainOnDev;
	protected Map<D, L> classifiedData;
	
	public ExperimentGST(String name, String inputPath, DataSet<D, L> trainData, DataSet<D, L> devData, DataSet<D, L> testData) {
		super(name, inputPath, trainData.getDatumTools());
		
		this.features = new ArrayList<Feature<D, L>>();
		this.gridSearchParameterValues = new HashMap<String, List<String>>();
		this.evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
	}
	
	@Override
	protected boolean execute() {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		FeaturizedDataSet<D, L> trainData = new FeaturizedDataSet<D, L>(this.name + " Training", this.features, this.maxThreads, this.datumTools, this.trainData.getLabelMapping());
		FeaturizedDataSet<D, L> devData = new FeaturizedDataSet<D, L>(this.name + " Dev", this.features, this.maxThreads, this.datumTools, this.devData.getLabelMapping());
		
		FeaturizedDataSet<D, L> testData = null;
		if (this.testData != null) {
			testData = new FeaturizedDataSet<D, L>(this.name + " Test", this.features, this.maxThreads, this.datumTools, this.testData.getLabelMapping());
			testData.addAll(this.testData);
		}
		
		trainData.addAll(this.trainData);
		devData.addAll(this.devData);
		
		output.debugWriteln("Initializing features (" + this.name + ")...");
		for (Feature<D, L> feature : this.features) {
			if (!feature.init(trainData))
				return false;
			
			trainData.addFeature(feature);
			devData.addFeature(feature);
			
			if (testData != null)
				testData.addFeature(feature);
		}
		
		GridSearchTestValidation<D, L> gridSearchValidation = new GridSearchTestValidation<D, L>(
				this.name, 
				this.model, 
				trainData, 
				devData, 
				testData, 
				this.evaluations,
				this.trainOnDev);
		gridSearchValidation.setPossibleHyperParameterValues(this.gridSearchParameterValues);
		if (gridSearchValidation.run(this.errorExampleExtractor, true, this.maxThreads).get(0) < 0){
			this.classifiedData = gridSearchValidation.getClassifiedData();
			return false;
		}
		this.classifiedData = gridSearchValidation.getClassifiedData();

		return true;
	}
	@Override
	protected boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.startsWith("model")) {
			String[] nameParts = nextName.split("_");
			String referenceName = null;
			if (nameParts.length > 1)
				referenceName = nameParts[1];
			
			String modelName = SerializationUtil.deserializeGenericName(reader);
			this.model = this.datumTools.makeModelInstance(modelName);
			if (!this.model.deserialize(reader, false, false, this.datumTools, referenceName))
				return false;
		} else if (nextName.startsWith("feature")) {
			String[] nameParts = nextName.split("_");
			String referenceName = null;
			boolean ignore = false;
			if (nameParts.length > 1)
				referenceName = nameParts[1];
			if (nameParts.length > 2)
				ignore = true;
			String featureName = SerializationUtil.deserializeGenericName(reader);
			Feature<D, L> feature = this.datumTools.makeFeatureInstance(featureName);
			if (!feature.deserialize(reader, false, false, this.datumTools, referenceName, ignore))
				return false;
			this.features.add(feature);
		} else if (nextName.startsWith("errorExampleExtractor")) {
			this.errorExampleExtractor = this.datumTools.getTokenSpanExtractor(
					SerializationUtil.deserializeAssignmentRight(reader));
			
		} else if (nextName.startsWith("gridSearchParameterValues")) {
			String parameterName = SerializationUtil.deserializeGenericName(reader);
			List<String> parameterValues = SerializationUtil.deserializeList(reader);
			this.gridSearchParameterValues.put(parameterName, parameterValues);
		
		} else if (nextName.startsWith("evaluation")) {
			String evaluationName = SerializationUtil.deserializeGenericName(reader);
			SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
			if (!evaluation.deserialize(reader, false, this.datumTools))
				return false;
			this.evaluations.add(evaluation);
			
		} else if (nextName.startsWith("trainOnDev")) {
			this.trainOnDev = Boolean.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		}
		
		return true;
	}
	
	public Map<D, L> getClassifiedData(){
		if (this.classifiedData == null)
			throw new IllegalStateException("Trying to return classified data, but the data hasn't been classified yet.");
		else
			return this.classifiedData;
	}
}
