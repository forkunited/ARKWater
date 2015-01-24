package ark.model.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.TokenSpanExtractor;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Timer;

/**
 * ValidationTrainTest trains a model on a training data
 * set and evaluates it on a test data set by given 
 * evaluation metrics
 * 
 * @author Bill McDowell
 *
 */
public class ValidationTrainTest<D extends Datum<L>, L> extends Validation<D, L> {
	private String name;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private Map<D, L> classifiedData;
	
	/**
	 * 
	 * @param name
	 * @param maxThreads
	 * @param model
	 * @param trainData 
	 * @param testData
	 * @param evaluations - Measures by which to evaluate the model
	 * @param errorExampleExtractor - Token span extractor used to generate error descriptions
	 *
	 */
	public ValidationTrainTest(String name,
							  int maxThreads,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> testData,
							  List<SupervisedModelEvaluation<D, L>> evaluations,
							  TokenSpanExtractor<D, L> errorExampleExtractor) {	
		super(name, trainData.getDatumTools(), maxThreads, model, evaluations, errorExampleExtractor);
		this.trainData = trainData;
		this.testData = testData;
	}
	
	public ValidationTrainTest(String name,
			  int maxThreads,
			  SupervisedModel<D, L> model, 
			  DataSet<D, L> trainData,
			  DataSet<D, L> testData,
			  List<SupervisedModelEvaluation<D, L>> evaluations,
			  TokenSpanExtractor<D, L> errorExampleExtractor) {	
		super(name, trainData.getDatumTools(), maxThreads, model, evaluations, errorExampleExtractor);
		this.trainData = new FeaturizedDataSet<D, L>(this.name + " Training", this.maxThreads, trainData.getDatumTools(), trainData.getLabelMapping());
		this.testData = new FeaturizedDataSet<D, L>(this.name + " Test", this.maxThreads, trainData.getDatumTools(), trainData.getLabelMapping());
	}
	
	public ValidationTrainTest(String name, Datum.Tools<D, L> datumTools, DataSet<D, L> trainData, DataSet<D, L> testData) {
		this(name, 1, null, trainData, testData, new ArrayList<SupervisedModelEvaluation<D, L>>(), null);
		
	}
	
	public ValidationTrainTest(String name, Datum.Tools<D, L> datumTools, FeaturizedDataSet<D, L> trainData, FeaturizedDataSet<D, L> testData) {
		this(name, 1, null, trainData, testData, new ArrayList<SupervisedModelEvaluation<D, L>>(), null);		
	}
	
	@Override
	public List<Double> run() {
		return run(false);
	}
	
	/**
	 * @param skipTraining indicates whether to skip model training
	 * @return trains and evaluates the model, returning a list of values as results
	 * of the evaluations
	 */
	public List<Double> run(boolean skipTraining) {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		Timer timer = this.trainData.getDatumTools().getDataTools().getTimer();
		
		this.evaluationValues = new ArrayList<Double>(this.evaluations.size());
		for (int i = 0; i < this.evaluations.size(); i++)
			this.evaluationValues.add(-1.0);
		
		
		if (!skipTraining) {
			timer.startClock(this.name + " Train/Test (Training)");
			output.debugWriteln("Training model (" + this.name + ")");
			
			if (!this.model.train(this.trainData, this.testData, this.evaluations))
				return this.evaluationValues;
			
			timer.stopClock(this.name + " Train/Test (Training)");
		}
		output.debugWriteln("Classifying data (" + this.name + ")");
		
		timer.startClock(this.name + " Train/Test (Testing)");
		Map<D, L> classifiedData = this.model.classify(this.testData);
		if (classifiedData == null)
			return this.evaluationValues;
		this.classifiedData = classifiedData;
		
		output.debugWriteln("Computing model score (" + this.name + ")");
		
		for (int i = 0; i < this.evaluations.size(); i++)
			this.evaluationValues.set(i, this.evaluations.get(i).evaluate(this.model, this.testData, classifiedData));
		
		this.confusionMatrix = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		this.confusionMatrix.addData(classifiedData);
		
		timer.stopClock(this.name + " Train/Test (Testing)");
		
		return this.evaluationValues;
	}
	
	public Map<D, L> getClassifiedData(){
		return this.classifiedData;
	}

	@Override
	public String[] getParameterNames() {
		return new String[0];
	}

	@Override
	public String getParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected boolean setMaxThreads(int maxThreads) {
		this.maxThreads = maxThreads;
		this.trainData.setMaxThreads(maxThreads);
		this.testData.setMaxThreads(maxThreads);
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
		
		if (!this.testData.addFeature(feature, false))
			return false;
		
		output.debugWriteln(this.name + " serializing feature (" + featureStr + ")...");
		output.modelWriteln(feature.toString(true));
		output.debugWriteln(this.name + " finished serializing feature (" + featureStr + ").");
		
		return true;
	}
}
