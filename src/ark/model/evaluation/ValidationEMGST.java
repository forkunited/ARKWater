package ark.model.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
import ark.util.ThreadMapper;
import ark.util.Timer;

public class ValidationEMGST<D extends Datum<L>, L> extends Validation<D, L> {
	private String[] parameters;
	private int iterations;
	private boolean firstIterationOnlyLabeled;
	private boolean relabelLabeledData;
	private List<SupervisedModelEvaluation<D, L>> unlabeledEvaluations;
	private List<GridSearch.GridDimension> gridDimensions; 
	
	private ValidationGST<D, L> validationGST;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> devData;
	private FeaturizedDataSet<D, L> testData;
	
	public ValidationEMGST(ValidationGST<D, L> validationGST,
			DataSet<D, L> trainData, 
			DataSet<D, L> devData,
			DataSet<D, L> testData,
			boolean relabelLabeledData) {
		super(validationGST.name, validationGST.datumTools);
		
		this.parameters = Arrays.copyOf(validationGST.parameters, validationGST.parameters.length + 2);
		this.parameters[this.parameters.length - 1] = "iterations";
		this.parameters[this.parameters.length - 2] = "firstIterationOnlyLabeled";
		
		this.validationGST = validationGST;
		this.trainData = new FeaturizedDataSet<D, L>(name + " Training", maxThreads, trainData.getDatumTools(), trainData.getLabelMapping()); 
		this.devData = new FeaturizedDataSet<D, L>(name + " Dev", maxThreads, devData.getDatumTools(), devData.getLabelMapping());
		this.testData = new FeaturizedDataSet<D, L>(name + " Test", maxThreads, testData.getDatumTools(), testData.getLabelMapping()); 
		this.unlabeledEvaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.gridDimensions = new ArrayList<GridSearch.GridDimension>();
		this.relabelLabeledData = relabelLabeledData;
		
		this.trainData.addAll(trainData);
		this.devData.addAll(devData);
		this.testData.addAll(testData);
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
		
		FeaturizedDataSet<D, L> labeledTrainData = (FeaturizedDataSet<D, L>)this.trainData.getSubset(DataFilter.OnlyLabeled);
		FeaturizedDataSet<D, L> labeledDevData = (FeaturizedDataSet<D, L>)this.devData.getSubset(DataFilter.OnlyLabeled);
		FeaturizedDataSet<D, L> labeledTestData = (FeaturizedDataSet<D, L>)this.testData.getSubset(DataFilter.OnlyLabeled);
		
		// First iteration only trains on labeled data?
		FeaturizedDataSet<D, L> trainData = (this.firstIterationOnlyLabeled) ? labeledTrainData : this.trainData;
		FeaturizedDataSet<D, L> devData = (this.firstIterationOnlyLabeled) ? labeledDevData : this.devData;
		FeaturizedDataSet<D, L> testData = (this.firstIterationOnlyLabeled) ? labeledTestData : this.testData;
		
		String initDebugFilePath = output.getDebugFilePath();
		String initResultsFilePath = output.getResultsFilePath();
		String initDataFilePath = output.getDataFilePath();
		String initModelFilePath = output.getModelFilePath();
		
		// Evaluate evaluation data with unlabeled evaluations (iteration 0)
		final FeaturizedDataSet<D, L> unlabeledEvaluationDataOnlyLabeled = labeledTestData;
		final Map<D, L> classifiedDataOnlyLabeled = new HashMap<D, L>();
		for (D datum : unlabeledEvaluationDataOnlyLabeled)
			classifiedDataOnlyLabeled.put(datum, datum.getLabel());
		ThreadMapper<SupervisedModelEvaluation<D, L>, Double> threadMapper = new ThreadMapper<SupervisedModelEvaluation<D, L>, Double>(
				new ThreadMapper.Fn<SupervisedModelEvaluation<D, L>, Double>() {
					@Override
					public Double apply(SupervisedModelEvaluation<D, L> evaluation) {
						return evaluation.evaluate(null, unlabeledEvaluationDataOnlyLabeled, classifiedDataOnlyLabeled);
					}
				});
		this.evaluationValues = threadMapper.run(this.unlabeledEvaluations, this.maxThreads);
		output.resultsWriteln("Iteration 0 (evaluated labeled data with unlabeled evaluations)");
		if (!outputResults())
			return null;
		//
		
		int i = 1;
		while (i <= this.iterations) {
			output.setDebugFile(new File(initDebugFilePath + "." + i), false);
			output.setResultsFile(new File(initResultsFilePath + "." + i), false);
			output.setDataFile(new File(initDataFilePath + "." + i), false);
			output.setModelFile(new File(initModelFilePath + "." + i), false);
			
			if (!this.validationGST.reset(trainData, devData, testData)
					|| this.validationGST.run() == null
					|| !this.validationGST.outputAll())
				return null;
		
			// Unlabeled evaluations
			final FeaturizedDataSet<D, L> unlabeledEvaluationData = this.testData;
			final Map<D, L> classifiedData = this.validationGST.getModel().classify(unlabeledEvaluationData);
			threadMapper = new ThreadMapper<SupervisedModelEvaluation<D, L>, Double>(
					new ThreadMapper.Fn<SupervisedModelEvaluation<D, L>, Double>() {
						@Override
						public Double apply(SupervisedModelEvaluation<D, L> evaluation) {
							return evaluation.evaluate(validationGST.getModel(), unlabeledEvaluationData, classifiedData);
						}
					});
			this.evaluationValues = threadMapper.run(this.unlabeledEvaluations, this.maxThreads);
			
			// Relabel training data
			Map<D, Map<L, Double>> trainP = this.validationGST.getModel().posterior(this.trainData);
			Map<D, L> trainC = this.validationGST.getModel().classify(this.trainData);
			for (Entry<D, Map<L, Double>> entry : trainP.entrySet()) {
				if (!this.relabelLabeledData && labeledTrainData.contains(entry.getKey())) {
					continue;
				}
				for (Entry<L, Double> labelEntry : entry.getValue().entrySet()) {
					entry.getKey().setLabelWeight(labelEntry.getKey(), labelEntry.getValue());
				}
				
				this.trainData.setDatumLabel(entry.getKey(), trainC.get(entry.getKey()));
			}
			
			output.setDebugFile(new File(initDebugFilePath), true);
			output.setResultsFile(new File(initResultsFilePath), true);
			output.setDataFile(new File(initDataFilePath), true);
			output.setModelFile(new File(initModelFilePath), true);
			output.resultsWriteln("Iteration " + i);
			if (!outputResults())
				return null;
				
			trainData = this.trainData;
			devData = labeledDevData;
			testData = labeledTestData;
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
		this.validationGST.gridDimensions = this.gridDimensions;
		
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
		else if (parameter.equals("firstIterationOnlyLabeled"))
			return String.valueOf(this.firstIterationOnlyLabeled);
		else 
			return this.validationGST.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		if (parameter.equals("iterations")) 
			this.iterations = Integer.valueOf(parameterValue);
		else if (parameter.equals("firstIterationOnlyLabeled"))
			this.firstIterationOnlyLabeled = Boolean.valueOf(parameterValue);
		else
			return this.validationGST.setParameterValue(parameter, parameterValue, datumTools);
		return true;
	}

	@Override
	protected boolean addFeature(Feature<D, L> feature) {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		Timer timer = this.datumTools.getDataTools().getTimer();
		String featureStr = feature.getReferenceName();
		
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
		this.validationGST.setMaxThreads(maxThreads);
		
		return true;
	}
	
	@Override
	public boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.equals("unlabeledEvaluation")) {
			String evaluationName = SerializationUtil.deserializeGenericName(reader);
			SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
			if (!evaluation.deserialize(reader, false, this.datumTools))
				return false;
			this.unlabeledEvaluations.add(evaluation);
		} else if (nextName.equals("gridDimension")) {
			GridSearch.GridDimension gridDimension = new GridSearch.GridDimension();
			if (!gridDimension.deserialize(reader))
				return false;
			
			this.gridDimensions.add(gridDimension);
		} else {
			return super.deserializeNext(reader, nextName);
		}
		
		return true;
	}
}
