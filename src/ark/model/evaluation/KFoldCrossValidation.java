package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.ClassificationEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

public class KFoldCrossValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private List<Feature<D, L>> features;
	private List<ClassificationEvaluation<D, L>> evaluations;
	private List<DataSet<D, L>> folds;
	private Map<String, List<String>> possibleParameterValues; // Hyper-parameter values
	private DecimalFormat cleanDouble;
	
	public KFoldCrossValidation(String name,
								SupervisedModel<D, L> model, 
								List<Feature<D, L>> features,
								List<ClassificationEvaluation<D, L>> evaluations,
								DataSet<D, L> data,
								int k,
								Random random) {
		this.name = name;
		this.model = model;
		this.features = features;
		this.evaluations = evaluations;
		
		double[] foldDistribution = new double[k];
		for (int i = 0; i < k; i++)
			foldDistribution[i] = 1.0/k;

		this.folds = data.makePartition(foldDistribution, random);
		this.possibleParameterValues = new HashMap<String, List<String>>();
		this.cleanDouble = new DecimalFormat("0.00");
	}
	
	public boolean addPossibleHyperParameterValue(String parameter, String parameterValue) {
		if (!this.possibleParameterValues.containsKey(parameter))
			this.possibleParameterValues.put(parameter, new ArrayList<String>());
		this.possibleParameterValues.get(parameter).add(parameterValue);
		
		return true;
	}
	
	public double run(int maxThreads, Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor) {
		ConfusionMatrix<D, L> aggregateConfusions = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		
		/*
		 * Get results, using a separate thread for each fold
		 */
		ExecutorService threadPool = Executors.newFixedThreadPool(maxThreads);
		List<ValidationThread> tasks = new ArrayList<ValidationThread>();
		List<ValidationResult> validationResults = new ArrayList<ValidationResult>(this.folds.size());
 		for (int i = 0; i < this.folds.size(); i++) {
 			validationResults.add(null);
			tasks.add(new ValidationThread(i, 1, errorExampleExtractor));
		}
		
		try {
			List<Future<ValidationResult>> results = threadPool.invokeAll(tasks);
			threadPool.shutdown();
			threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			for (Future<ValidationResult> futureResult : results) {
				ValidationResult result = futureResult.get();
				if (result == null)
					return -1.0;
				validationResults.set(result.getFoldIndex(), result);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return -1.0;
		}
		
		/* 
		 * Output results for each fold
		 */
		OutputWriter output = this.folds.get(0).getDatumTools().getDataTools().getOutputWriter();
		String gridSearchParameters = ((this.possibleParameterValues.size() > 0) ? validationResults.get(0).getBestParameters().toKeyString("\t") + "\t" : "");
		String evaluationsStr = "";
		List<Double> averageEvaluations = new ArrayList<Double>(this.evaluations.size());
		for (int i = 0; i < this.evaluations.size(); i++) {
			evaluationsStr += this.evaluations.get(i).toString() + "\t";
			averageEvaluations.add(0.0);
		}
		output.resultsWriteln("Fold\t" + gridSearchParameters + "\t" + evaluationsStr);
		
		for (int i = 0; i < validationResults.size(); i++) {
			String gridSearchParameterValues = ((this.possibleParameterValues.size() > 0) ? validationResults.get(i).getBestParameters().toValueString("\t") + "\t" : "");
			List<Double> evaluationValues = validationResults.get(i).getEvaluationValues();
			String evaluationValuesStr = "";
			for (int j = 0; j < evaluationValues.size(); j++) {
				evaluationValuesStr += this.cleanDouble.format(evaluationValues.get(j)) + "\t";
				averageEvaluations.set(j , averageEvaluations.get(j) + evaluationValues.get(j));
			}
			
			output.resultsWriteln(i + "\t" + gridSearchParameterValues + evaluationValuesStr);
			aggregateConfusions.add(validationResults.get(i).getConfusionMatrix());
		}
		
		output.resultsWrite("Averages:\t");
		for (int i = 0; i < averageEvaluations.size(); i++) {
			averageEvaluations.set(i, averageEvaluations.get(i)/this.folds.size());
			output.resultsWrite(this.cleanDouble.format(averageEvaluations.get(i)) + "\t");
		}
		output.resultsWriteln("");
		
		/*
		 * Output confusion matrix
		 */
		output.resultsWriteln("\nTotal Confusion Matrix:\n " + aggregateConfusions.toString());
		
		/*
		 * Output grid search results
		 */
		if (this.possibleParameterValues.size() > 0) {
			output.resultsWriteln("\nGrid search results:");
			output.resultsWrite(validationResults.get(0).getGridEvaluation().get(0).getFirst().toKeyString("\t") + "\t");
			for (int i = 0; i < this.folds.size(); i++)
				output.resultsWrite("Fold " + i + "\t");
			output.resultsWrite("\n");
			
			List<Pair<HyperParameterGridSearch.GridPosition, List<Double>>> gridFoldResults = new ArrayList<Pair<HyperParameterGridSearch.GridPosition, List<Double>>>();
			for (int i = 0; i < validationResults.size(); i++) {
				List<Pair<HyperParameterGridSearch.GridPosition, Double>> gridEvaluation = validationResults.get(i).getGridEvaluation();
				for (int j = 0; j < gridEvaluation.size(); j++) {
					if (gridFoldResults.size() <= j)
						gridFoldResults.add(new Pair<HyperParameterGridSearch.GridPosition, List<Double>>(gridEvaluation.get(j).getFirst(), new ArrayList<Double>()));
					gridFoldResults.get(j).getSecond().add(gridEvaluation.get(j).getSecond());
				}
			}
			
			for (Pair<HyperParameterGridSearch.GridPosition, List<Double>> gridFoldResult : gridFoldResults) {
				output.resultsWrite(gridFoldResult.getFirst().toValueString("\t") + "\t");
				for (int i = 0; i < gridFoldResult.getSecond().size(); i++) {
					output.resultsWrite(this.cleanDouble.format(gridFoldResult.getSecond().get(i)) + "\t");
				}
				output.resultsWrite("\n");
			}
		}
		
		return averageEvaluations.get(0);
	}
	
	private class ValidationResult  {
		private int foldIndex;
		private List<Double> evaluationValues;
		private ConfusionMatrix<D, L> confusionMatrix;
		private List<Pair<HyperParameterGridSearch.GridPosition, Double>> gridEvaluation;
		private HyperParameterGridSearch.GridPosition bestParameters;
		
		public ValidationResult(int foldIndex, List<Double> evaluationValues, ConfusionMatrix<D, L> confusionMatrix, List<Pair<HyperParameterGridSearch.GridPosition, Double>> gridEvaluation, HyperParameterGridSearch.GridPosition bestParameters) {
			this.foldIndex = foldIndex;
			this.evaluationValues = evaluationValues;
			this.confusionMatrix = confusionMatrix;
			this.gridEvaluation = gridEvaluation;
			this.bestParameters = bestParameters;
		}
		
		public int getFoldIndex() {
			return this.foldIndex;
		}
		
		public List<Double> getEvaluationValues() {
			return this.evaluationValues;
		}
		
		public ConfusionMatrix<D, L> getConfusionMatrix() {
			return this.confusionMatrix;
		}
		
		public List<Pair<HyperParameterGridSearch.GridPosition, Double>> getGridEvaluation() {
			return this.gridEvaluation;
		}
		
		public HyperParameterGridSearch.GridPosition getBestParameters() {
			return this.bestParameters;
		}
	}
	
	private class ValidationThread implements Callable<ValidationResult> {
		private int foldIndex;
		private int maxThreads;
		private Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor;
		private Map<String, String> parameterEnvironment;
		
		public ValidationThread(int foldIndex, int maxThreads, Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor) {
			this.foldIndex = foldIndex;
			this.maxThreads = maxThreads;
			this.errorExampleExtractor = errorExampleExtractor;
	
			this.parameterEnvironment = new HashMap<String, String>();
			this.parameterEnvironment.putAll(folds.get(foldIndex).getDatumTools().getDataTools().getParameterEnvironment());
			this.parameterEnvironment.put("FOLD", String.valueOf(this.foldIndex));
		}
		
		public ValidationResult call() {
			OutputWriter output = folds.get(foldIndex).getDatumTools().getDataTools().getOutputWriter();
			String namePrefix = name + " Fold " + foldIndex;
			
			/*
			 * Initialize training, dev, and test sets for each fold
			 */
			output.debugWriteln("Initializing CV data sets for " + name);
			Datum.Tools<D, L> datumTools = folds.get(this.foldIndex).getDatumTools();
			Datum.Tools.LabelMapping<L> labelMapping = folds.get(this.foldIndex).getLabelMapping();
			FeaturizedDataSet<D, L> testData = new FeaturizedDataSet<D, L>(namePrefix + " Test", features, this.maxThreads, datumTools, labelMapping);
			FeaturizedDataSet<D, L> trainData = new FeaturizedDataSet<D, L>(namePrefix + " Training", features, this.maxThreads, datumTools, labelMapping);
			FeaturizedDataSet<D, L> devData = new FeaturizedDataSet<D, L>(namePrefix + " Dev", features, this.maxThreads, datumTools, labelMapping);
			for (int j = 0; j < folds.size(); j++) {
				if (j == this.foldIndex) {
					testData.addAll(folds.get(j));
				} else if (possibleParameterValues.size() > 0 && j == ((foldIndex + 1) % folds.size())) {
					devData.addAll(folds.get(j));
				} else {
					trainData.addAll(folds.get(j));	
				}
			}
			
			/* Need cloned bunch of features for each fold so that they can be 
			 * reinitialized for each training set */
			output.debugWriteln("Initializing features for CV fold " + this.foldIndex);
			for (Feature<D, L> feature : features) {
				Feature<D, L> foldFeature = feature.clone(datumTools, this.parameterEnvironment);
				if (!foldFeature.init(trainData))
					return null;
				
				trainData.addFeature(foldFeature);
				devData.addFeature(foldFeature);
				testData.addFeature(foldFeature);
			}
			
			/*
			 * Perform a grid search using the training and dev data for this fold
			 */
			SupervisedModel<D, L> foldModel = model.clone(datumTools, this.parameterEnvironment);
			List<Pair<HyperParameterGridSearch.GridPosition, Double>> gridEvaluation = null;
			HyperParameterGridSearch.GridPosition bestParameters = null;
			if (possibleParameterValues.size() > 0) {
				HyperParameterGridSearch<D, L> gridSearch = 
						new HyperParameterGridSearch<D,L>(namePrefix,
														  foldModel,
										 				  trainData, 
										 				  devData,
										 				  possibleParameterValues,
										 				  evaluations.get(0)); 
				bestParameters = gridSearch.getBestPosition();
				gridEvaluation = gridSearch.getGridEvaluation();
				
				output.debugWriteln("Grid search on fold " + foldIndex + ": \n" + gridSearch.toString());
				
				if (bestParameters != null)
					foldModel.setHyperParameterValues(bestParameters.getCoordinates(), datumTools);
			}
			
			/*
			 * Train the model using the best hyper-parameters from the grid search
			 */
			output.debugWriteln("Training model for CV fold " + this.foldIndex);
			TrainTestValidation<D, L> accuracy = new TrainTestValidation<D, L>(namePrefix, foldModel,  trainData, testData, evaluations);
			List<Double> evaluationValues = accuracy.run();
			if (evaluationValues.get(0) < 0) {
				output.debugWriteln("Error: Validation failed on fold " + this.foldIndex);
				return new ValidationResult(foldIndex, evaluationValues, null, null, null);
			} else {
				ConfusionMatrix<D, L> confusions = accuracy.getConfusionMatrix();
				output.debugWriteln(evaluations.get(0).toString() + " on fold " + this.foldIndex + ": " + cleanDouble.format(evaluationValues.get(0)));
				
				output.dataWriteln("--------------- Fold: " + this.foldIndex + " ---------------");
				output.dataWriteln(confusions.getActualToPredictedDescription(this.errorExampleExtractor));
				
				output.modelWriteln("--------------- Fold: " + this.foldIndex + " ---------------");
				output.modelWriteln(foldModel.toString());
				
				return new ValidationResult(foldIndex, evaluationValues, accuracy.getConfusionMatrix(), gridEvaluation, bestParameters);
			}
		}
	}
}