package ark.model.evaluation;

import java.io.BufferedReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
 * ValidationGST (grid-search-test) performs a grid search for model
 * hyper-parameter values using a training and dev data sets.
 * Then, it sets the hyper-parameters to the best values from
 * the grid search, retrains on the train+dev data, and
 * gives a final evaluation on the test data. 
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class ValidationGST<D extends Datum<L>, L> extends Validation<D, L> {
	protected FeaturizedDataSet<D, L> trainData;
	protected FeaturizedDataSet<D, L> devData;
	protected FeaturizedDataSet<D, L> testData;
	protected List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation;
	protected GridSearch<D, L>.EvaluatedGridPosition bestGridPosition;
	protected List<GridSearch.GridDimension> gridDimensions; 
	
	protected boolean trainOnDev;
	protected String[] parameters = new String[] { "trainOnDev" };
	
	public ValidationGST(String name,
						  int maxThreads,
						  SupervisedModel<D, L> model, 
						  FeaturizedDataSet<D, L> trainData,
						  FeaturizedDataSet<D, L> devData,
						  FeaturizedDataSet<D, L> testData,
						  List<SupervisedModelEvaluation<D, L>> evaluations,
						  TokenSpanExtractor<D, L> errorExampleExtractor,
						  List<GridSearch.GridDimension> gridDimensions,
						  boolean trainOnDev) {
		super(name, trainData.getDatumTools(), maxThreads, model, evaluations, errorExampleExtractor);
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
		this.gridDimensions = gridDimensions;
		this.trainOnDev = trainOnDev;
		this.gridEvaluation = new ArrayList<GridSearch<D, L>.EvaluatedGridPosition>();
	}
	
	public ValidationGST(String name,
			  int maxThreads,
			  SupervisedModel<D, L> model, 
			  List<Feature<D, L>> features,
			  DataSet<D, L> trainData,
			  DataSet<D, L> devData,
			  DataSet<D, L> testData,
			  List<SupervisedModelEvaluation<D, L>> evaluations,
			  TokenSpanExtractor<D, L> errorExampleExtractor,
			  List<GridSearch.GridDimension> gridDimensions,
			  boolean trainOnDev) {
		this(name, 
			 maxThreads, 
			 model,
			 new FeaturizedDataSet<D, L>(name + " Training", maxThreads, trainData.getDatumTools(), trainData.getLabelMapping()), 
			 new FeaturizedDataSet<D, L>(name + " Dev", maxThreads, devData.getDatumTools(), devData.getLabelMapping()), 
			 (testData == null) ? null : new FeaturizedDataSet<D, L>(name + " Test", maxThreads, testData.getDatumTools(), testData.getLabelMapping()), 
			 evaluations,
			 errorExampleExtractor,
			 gridDimensions,
			 trainOnDev);
		
		this.trainData.addAll(trainData);
		this.devData.addAll(devData);
		if (this.testData != null)
			this.testData.addAll(testData);
		
		for (Feature<D, L> feature : features)
			addFeature(feature);
	}
	
	public ValidationGST(String name, 
						 Datum.Tools<D, L> datumTools, 
						 DataSet<D, L> trainData, 
						 DataSet<D, L> devData, 
						 DataSet<D, L> testData) {
		this(name, 
				1, 
				null, 
				new ArrayList<Feature<D, L>>(),
				trainData, 
				devData, 
				testData, 
				new ArrayList<SupervisedModelEvaluation<D, L>>(), 
				null, 
				new ArrayList<GridSearch.GridDimension>(),
				false);
		
	}
	
	public ValidationGST(String name, 
			 Datum.Tools<D, L> datumTools, 
			 FeaturizedDataSet<D, L> trainData, 
			 FeaturizedDataSet<D, L> devData, 
			 FeaturizedDataSet<D, L> testData) {
		this(name, 
			1, 
			null, 
			trainData, 
			devData, 
			testData, 
			new ArrayList<SupervisedModelEvaluation<D, L>>(), 
			null, 
			new ArrayList<GridSearch.GridDimension>(), 
			false);	
	}
	
	public ValidationGST(String name, Datum.Tools<D, L> datumTools) {
		super(name, datumTools);
	}
	
	public List<GridSearch<D, L>.EvaluatedGridPosition> getGridEvaluation() {
		return this.gridEvaluation;
	}
	
	public GridSearch<D, L>.EvaluatedGridPosition getBestGridPosition() {
		return this.bestGridPosition;
	}
	
	public boolean reset(FeaturizedDataSet<D, L> trainData, FeaturizedDataSet<D, L> devData, FeaturizedDataSet<D, L> testData) {
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
		
		this.evaluationValues = new ArrayList<Double>();
		this.confusionMatrix = null;
		this.gridEvaluation = null;
		this.bestGridPosition = null;
		
		return true;
	}
	
	@Override
	public List<Double> run() {
		Timer timer = this.datumTools.getDataTools().getTimer();
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		DecimalFormat cleanDouble = new DecimalFormat("0.00000");
		
		timer.startClock(this.name + " GST (Total)");
		timer.startClock(this.name + " Grid Search");
		
		if (this.gridDimensions.size() > 0) {
			GridSearch<D, L> gridSearch = new GridSearch<D,L>(this.name,
										this.model,
						 				this.trainData, 
						 				this.devData,
						 				this.gridDimensions,
						 				this.evaluations.get(0)); 
			this.bestGridPosition = gridSearch.getBestPosition(this.maxThreads);
			this.gridEvaluation = gridSearch.getGridEvaluation(this.maxThreads);
				
			output.debugWriteln("Grid search (" + this.name + "): \n" + gridSearch.toString());
				
			if (this.bestGridPosition != null) {
				if (!this.trainOnDev) 
					this.model = this.bestGridPosition.getValidation().getModel();
				else {
					this.model.setHyperParameterValues(this.bestGridPosition.getCoordinates(), this.trainData.getDatumTools());
					this.model = this.model.clone(this.trainData.getDatumTools());
				}
			}
		}
		
		timer.stopClock(this.name + " Grid Search");
		
		
		output.debugWriteln("Train and/or evaluating model with best parameters (" + this.name + ")");
		
		this.evaluationValues = null;
		if (this.testData != null) {
			if (this.trainOnDev) 
				this.trainData.addAll(this.devData); // FIXME Reinitialize features on train+dev?
	
			ValidationTrainTest<D, L> accuracy = new ValidationTrainTest<D, L>(this.name, 1, this.model, this.trainData, this.testData, this.evaluations, this.errorExampleExtractor);
			this.evaluationValues = accuracy.run(!this.trainOnDev);
			if (this.evaluationValues.get(0) < 0) {
				output.debugWriteln("Error: Validation failed (" + this.name + ")");
				return null;
			} 
			
			this.confusionMatrix = accuracy.getConfusionMatrix();
			output.debugWriteln("Test " + this.evaluations.get(0).toString() + " (" + this.name + "): " + cleanDouble.format(this.evaluationValues.get(0)));
			
		} else {
			this.evaluationValues = this.bestGridPosition.getValidation().getEvaluationValues();
			this.confusionMatrix = this.bestGridPosition.getValidation().getConfusionMatrix();
			output.debugWriteln("Dev best " + this.evaluations.get(0).toString() + " (" + this.name + "): " + cleanDouble.format(this.evaluationValues.get(0)));
			this.model = this.bestGridPosition.getValidation().getModel();
		}
		
		timer.stopClock(this.name + " GST (Total)");
		
		return evaluationValues;
	}
	
	public boolean outputResults() {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		
		if (this.bestGridPosition != null) {
			Map<String, String> parameters = this.bestGridPosition.getCoordinates();
			output.resultsWriteln("Best parameters from grid search:");
			for (Entry<String, String> entry : parameters.entrySet())
				output.resultsWriteln(entry.getKey() + ": " + entry.getValue());
		}
		
		if (this.testData != null) {
			output.resultsWriteln("\nTest set evaluation results: ");
		} else {
			output.resultsWriteln("\nDev set best evaluation results: ");
		}
		
		for (int i = 0; i < this.evaluations.size(); i++)
			output.resultsWriteln(this.evaluations.get(i).toString() + ": " + evaluationValues.get(i));
		
		output.resultsWriteln("\nConfusion matrix:\n" + this.confusionMatrix.toString());
		
		if (this.gridEvaluation != null && this.gridEvaluation.size() > 0) {
			output.resultsWriteln("\nGrid search on " + this.evaluations.get(0).toString() + ":");
			output.resultsWriteln(this.gridEvaluation.get(0).toKeyString("\t") + "\t" + this.evaluations.get(0).toString());
			for (GridSearch<D, L>.EvaluatedGridPosition gridPosition : this.gridEvaluation) {
				output.resultsWriteln(gridPosition.toValueString("\t") + "\t" + gridPosition.getPositionValue());
			}
		}		
	
		return true;
	}
	
	@Override
	protected boolean setMaxThreads(int maxThreads) {
		this.maxThreads = maxThreads;
		if (this.trainData != null)
			this.trainData.setMaxThreads(maxThreads);
		if (this.trainData != null)
			this.devData.setMaxThreads(maxThreads);
		if (this.testData != null)
			this.testData.setMaxThreads(maxThreads);
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
	
		if (!this.devData.addFeature(feature, false) || (this.testData != null && !this.testData.addFeature(feature, false)))
			return false;
		
		output.debugWriteln(this.name + " serializing feature (" + featureStr + ")...");
		output.modelWriteln(feature.toString(true));
		output.debugWriteln(this.name + " finished serializing feature (" + featureStr + ").");
		
		return true;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameters;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("trainOnDev"))
			return String.valueOf(this.trainOnDev);
		
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		if (parameter.equals("trainOnDev")) {
			this.trainOnDev = Boolean.valueOf(parameterValue);
		} else
			return false;
		
		return true;
	}
	
	@Override
	public boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.equals("gridDimension")) {
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
