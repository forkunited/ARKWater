package ark.model.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.DataSet;
import ark.data.annotation.DataSet.DataFilter;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.SerializationUtil;
import ark.util.Timer;

public class ValidationEMGST<D extends Datum<L>, L> extends Validation<D, L> {
	private String[] parameters;
	private int iterations;
	private List<SupervisedModelEvaluation<D, L>> unlabeledEvaluations;
	
	private ValidationGST<D, L> validationGST;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> devData;
	private FeaturizedDataSet<D, L> testData;
	
	public ValidationEMGST(ValidationGST<D, L> validationGST,
			DataSet<D, L> trainData, 
			DataSet<D, L> devData,
			DataSet<D, L> testData) {
		super(validationGST.name, validationGST.datumTools);
		
		this.parameters = Arrays.copyOf(validationGST.parameters, this.parameters.length + 2);
		this.parameters[this.parameters.length - 1] = "iterations";
		this.parameters[this.parameters.length - 2] = "unlabeledEvaluation";
		
		this.validationGST = validationGST;
		this.trainData = new FeaturizedDataSet<D, L>(name + " Training", maxThreads, trainData.getDatumTools(), trainData.getLabelMapping()); 
		this.devData = new FeaturizedDataSet<D, L>(name + " Dev", maxThreads, devData.getDatumTools(), devData.getLabelMapping());
		this.testData = new FeaturizedDataSet<D, L>(name + " Test", maxThreads, testData.getDatumTools(), testData.getLabelMapping()); 
		this.unlabeledEvaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
	}

	@Override
	public List<Double> run() {
		Timer timer = this.trainData.getDatumTools().getDataTools().getTimer();
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		if (!copyIntoValidationGST())
			return null;
		
		timer.startClock(this.name + " Feature Computation");
		output.debugWriteln("Computing features...");
		if (!this.trainData.precomputeFeatures()
				|| !this.devData.precomputeFeatures()
				|| (this.testData != null && !this.testData.precomputeFeatures()))
			return null;
		output.debugWriteln("Finished computing features.");
		timer.stopClock(this.name + " Feature Computation");
		
		this.evaluationValues = new ArrayList<Double>();
		for (int i = 0; i < this.unlabeledEvaluations.size(); i++)
			this.evaluationValues.add(0.0);
		
		// First iteration only trains on labeled data
		FeaturizedDataSet<D, L> trainData = (FeaturizedDataSet<D, L>)this.trainData.getSubset(DataFilter.OnlyLabeled);
		FeaturizedDataSet<D, L> devData = (FeaturizedDataSet<D, L>)this.devData.getSubset(DataFilter.OnlyLabeled);
		FeaturizedDataSet<D, L> testData = (FeaturizedDataSet<D, L>)this.testData.getSubset(DataFilter.OnlyLabeled);
		
		int i = 1;
		while (i < this.iterations) {
			output.setDebugFile(new File(output.getDebugFilePath() + "." + i));
			output.setResultsFile(new File(output.getResultsFilePath() + "." + i));
			output.setDataFile(new File(output.getDataFilePath() + "." + i));
			output.setModelFile(new File(output.getModelFilePath() + "." + i));
			
			if (!this.validationGST.reset(trainData, devData, testData)
					|| this.validationGST.run() == null
					|| !this.validationGST.outputAll())
				return null;
		
			// Unlabeled evaluations
			Map<D, L> classifiedData = this.validationGST.getModel().classify(this.testData);
			for (int j = 0; j < this.unlabeledEvaluations.size(); j++)
				this.evaluationValues.set(j, this.unlabeledEvaluations.get(j).evaluate(this.model, this.testData, classifiedData));
			
			// Relabel training data
			Map<D, Map<L, Double>> trainP = this.validationGST.getModel().posterior(this.trainData);
			for (Entry<D, Map<L, Double>> entry : trainP.entrySet()) {
				L maxLabel = null;
				double maxLabelWeight = Double.NEGATIVE_INFINITY;
				for (Entry<L, Double> labelEntry : entry.getValue().entrySet()) {
					if (labelEntry.getValue() > maxLabelWeight) {
						maxLabel = labelEntry.getKey();
						maxLabelWeight = labelEntry.getValue();
					}
					entry.getKey().setLabelWeight(labelEntry.getKey(), labelEntry.getValue());
				}
				
				entry.getKey().setLabel(maxLabel);
			}
			
			if (!outputResults())
				return null;
				
			trainData = this.trainData;
			devData = this.devData;
			testData = this.testData;
			i++;
		}
		
		this.model = this.validationGST.getModel();
		this.evaluationValues = this.validationGST.evaluationValues;
		this.confusionMatrix = this.validationGST.confusionMatrix;
		
		return this.evaluationValues;
	}
	
	private boolean copyIntoValidationGST() {
		this.validationGST.model = this.model;
		this.validationGST.errorExampleExtractor = this.errorExampleExtractor;
		this.validationGST.evaluations = this.evaluations;
		
		return true;
	}
	
	@Override
	public boolean outputResults() {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		
		output.resultsWriteln("\nUnlabeled data evaluation results: ");
		for (int i = 0; i < this.unlabeledEvaluations.size(); i++)
			output.resultsWriteln(this.unlabeledEvaluations.get(i).toString() + ": " + this.evaluationValues.get(i));
		
		return true;
	}
	
	@Override
	public boolean outputModel() {
		return this.validationGST.outputModel();
	}
	
	@Override
	public boolean outputData() {
		return this.validationGST.outputData();
	}
	
	@Override
	public String[] getParameterNames() {
		return this.parameters;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("iterations"))
			return String.valueOf(this.iterations);
		else if (parameter.equals("unlabeledEvaluation")) {
			return "[...]";
		} else 
			return this.validationGST.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		if (parameter.equals("iterations")) 
			this.iterations = Integer.valueOf(parameterValue);
		else if (parameter.equals("unlabeledEvaluation")) {
			// FIXME: Hack
			BufferedReader reader = new BufferedReader(new StringReader(parameterValue + ")"));
			try {
				String evaluationName = SerializationUtil.deserializeGenericName(reader);
				SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
				if (!evaluation.deserialize(reader, false, this.datumTools))
					return false;
				this.unlabeledEvaluations.add(evaluation);
			} catch (IOException e) {
				return false;
			}
		} else
			return this.validationGST.setParameterValue(parameter, parameterValue, datumTools);
		return true;
	}

	@Override
	protected boolean addFeature(Feature<D, L> feature) {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		Timer timer = this.datumTools.getDataTools().getTimer();
		String featureStr = feature.toString(false);
		
		output.debugWriteln(this.name + " initializing feature (" + featureStr + ")...");
		timer.startClock(featureStr + " Initialization");
		if (!this.trainData.addFeature(feature, true))
			return false;
		timer.stopClock(featureStr + " Initialization");
		output.debugWriteln(this.name + " finished initializing feature (" + featureStr + ").");
	
		if (!this.devData.addFeature(feature, false) || !this.testData.addFeature(feature, false))
			return false;
		
		output.debugWriteln(this.name + " serializing feature (" + featureStr + ")...");
		output.modelWriteln(feature.toString(true));
		output.debugWriteln(this.name + " finished serializing feature (" + featureStr + ").");
		
		return true;
	}

	@Override
	protected boolean setMaxThreads(int maxThreads) {
		this.maxThreads = maxThreads;
		this.trainData.setMaxThreads(maxThreads);
		this.devData.setMaxThreads(maxThreads);
		this.testData.setMaxThreads(maxThreads);
		
		return true;
	}
}
