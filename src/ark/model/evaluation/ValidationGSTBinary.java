package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.Datum.Tools.TokenSpanExtractor;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCompositeBinary;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.ThreadMapper;
import ark.util.ThreadMapper.Fn;
import ark.util.Timer;

/**
 * ValidationGSTBinary performs ValidationGST validations on several
 * binary models in parallel.  There is a separate validation for
 * each label indicator stored in the given Datum.Tools object.
 * 
 * @author Bill McDowell
 *
 * @param <T>
 * @param <D>
 * @param <L>
 */
public class ValidationGSTBinary<T extends Datum<Boolean>, D extends Datum<L>, L> extends ValidationGST<D, L> {
	private ConfusionMatrix<T, Boolean> aggregateConfusionMatrix;
	private List<ValidationGST<T, Boolean>> binaryValidations;
	private Map<GridSearch<T, Boolean>.GridPosition, Integer> bestPositionCounts;
	private SupervisedModelCompositeBinary<T, D, L> learnedCompositeModel;
	
	private Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator;
	
	public ValidationGSTBinary(String name,
			  int maxThreads,
			  SupervisedModel<D, L> model, 
			  FeaturizedDataSet<D, L> trainData,
			  FeaturizedDataSet<D, L> devData,
			  FeaturizedDataSet<D, L> testData,
			  List<SupervisedModelEvaluation<D, L>> evaluations,
			  TokenSpanExtractor<D, L> errorExampleExtractor,
			  Map<String, List<String>> possibleParameterValues,
			  boolean trainOnDev,
			  Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, 
			  maxThreads,
			  model,
			  trainData,
			  devData,
			  testData,
			  evaluations,
			  errorExampleExtractor,
			  possibleParameterValues,
			  trainOnDev);
	
		this.parameters = Arrays.copyOf(this.parameters, this.parameters.length + 1);
		this.parameters[this.parameters.length - 1] = "inverseLabelIndicator";
		this.inverseLabelIndicator = inverseLabelIndicator;
	}

	public ValidationGSTBinary(String name,
							   int maxThreads,
							   SupervisedModel<D, L> model, 
							   List<Feature<D, L>> features,
							   DataSet<D, L> trainData,
							   DataSet<D, L> devData,
							   DataSet<D, L> testData,
							   List<SupervisedModelEvaluation<D, L>> evaluations,
							   TokenSpanExtractor<D, L> errorExampleExtractor,
							   Map<String, List<String>> possibleParameterValues,
							   boolean trainOnDev,
							   Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, 
			  maxThreads,
			  model,
			  features,
			  trainData,
			  devData,
			  testData,
			  evaluations,
			  errorExampleExtractor,
			  possibleParameterValues,
			  trainOnDev);
		
		this.parameters = Arrays.copyOf(this.parameters, this.parameters.length + 1);
		this.parameters[this.parameters.length - 1] = "inverseLabelIndicator";
		this.inverseLabelIndicator = inverseLabelIndicator;
	}

	public ValidationGSTBinary(String name, 
				 Datum.Tools<D, L> datumTools, 
				 DataSet<D, L> trainData, 
				 DataSet<D, L> devData, 
				 DataSet<D, L> testData) {
		super(name, datumTools, trainData, devData, testData);
		this.parameters = Arrays.copyOf(this.parameters, this.parameters.length + 1);
		this.parameters[this.parameters.length - 1] = "inverseLabelIndicator";
	}

	public ValidationGSTBinary(String name, 
						 Datum.Tools<D, L> datumTools, 
						 FeaturizedDataSet<D, L> trainData, 
						 FeaturizedDataSet<D, L> devData, 
						 FeaturizedDataSet<D, L> testData) {
		super(name, datumTools, trainData, devData, testData);
		this.parameters = Arrays.copyOf(this.parameters, this.parameters.length + 1);
		this.parameters[this.parameters.length - 1] = "inverseLabelIndicator";
	}
	
	public ValidationGSTBinary(String name, Datum.Tools<D, L> datumTools) {
		super(name, datumTools);
		this.parameters = Arrays.copyOf(this.parameters, this.parameters.length + 1);
		this.parameters[this.parameters.length - 1] = "inverseLabelIndicator";
	}
	
	@Override
	public boolean reset(FeaturizedDataSet<D, L> trainData, FeaturizedDataSet<D, L> devData, FeaturizedDataSet<D, L> testData) {
		if (!super.reset(trainData, devData, testData))
			return false;
	
		this.aggregateConfusionMatrix = null;
		this.binaryValidations = null;
		this.bestPositionCounts = null;
		this.learnedCompositeModel = null;
		
		return true;
	}
	
	@Override 
	public SupervisedModel<D, L> getModel() {
		return this.learnedCompositeModel;
	}
	
	@Override
	public List<Double> run() { 
		Timer timer = this.trainData.getDatumTools().getDataTools().getTimer();
		final OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		final Set<Boolean> binaryLabels = new HashSet<Boolean>();
		binaryLabels.add(true);
		binaryLabels.add(false);
		
		if (!this.trainData.getPrecomputedFeatures() 
				|| !this.devData.getPrecomputedFeatures() 
				|| (this.testData != null && this.testData.getPrecomputedFeatures())) {
			timer.startClock(this.name + " Feature Computation");
			output.debugWriteln("Computing features...");
			if (!this.trainData.precomputeFeatures()
					|| !this.devData.precomputeFeatures()
					|| (this.testData != null && !this.testData.precomputeFeatures()))
				return null;
			output.debugWriteln("Finished computing features.");
			timer.stopClock(this.name + " Feature Computation");
		}
		output.resultsWriteln("Training data examples: " + this.trainData.size());
		output.resultsWriteln("Dev data examples: " + this.devData.size());
		output.resultsWriteln("Test data examples: " + this.testData.size());
		output.resultsWriteln("Feature vocabulary size: " + this.trainData.getFeatureVocabularySize() + "\n");
		
		ThreadMapper<LabelIndicator<L>, ValidationGST<T, Boolean>> threads = new ThreadMapper<LabelIndicator<L>, ValidationGST<T, Boolean>>(new Fn<LabelIndicator<L>, ValidationGST<T, Boolean>>() {
			public ValidationGST<T, Boolean> apply(LabelIndicator<L> labelIndicator) {
				Datum.Tools<T, Boolean> binaryTools = trainData.getDatumTools().makeBinaryDatumTools(labelIndicator);
				SupervisedModel<T, Boolean> binaryModel = model.clone(binaryTools, binaryTools.getDataTools().getParameterEnvironment(), false);
				
				binaryModel.setLabels(binaryLabels, null);
				
				List<SupervisedModelEvaluation<T, Boolean>> binaryEvaluations = new ArrayList<SupervisedModelEvaluation<T, Boolean>>();
				for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
					binaryEvaluations.add(evaluation.clone(binaryTools, binaryTools.getDataTools().getParameterEnvironment(), false));
				}
				
				FeaturizedDataSet<T, Boolean> binaryTrainData = (FeaturizedDataSet<T, Boolean>)trainData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				FeaturizedDataSet<T, Boolean> binaryDevData = (FeaturizedDataSet<T, Boolean>)devData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				FeaturizedDataSet<T, Boolean> binaryTestData = (testData == null) ? null : (FeaturizedDataSet<T, Boolean>)testData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				
				ValidationGST<T, Boolean> binaryValidation = new ValidationGST<T, Boolean>(
						labelIndicator.toString(),
						(int)Math.ceil(maxThreads / (double)trainData.getDatumTools().getLabelIndicators().size()),
						binaryModel,
						binaryTrainData,
						binaryDevData,
						binaryTestData,			
						binaryEvaluations,
						binaryTools.getTokenSpanExtractor(errorExampleExtractor.toString()),
						possibleParameterValues,
						trainOnDev);
				
				if (binaryTrainData.getDataSizeForLabel(true)/(double)binaryTrainData.size() <= 0.0001
						|| binaryDevData.getDataSizeForLabel(true) == 0 
						|| (binaryTestData != null && binaryTestData.getDataSizeForLabel(true) == 0)) {
					output.debugWriteln("Skipping " + labelIndicator.toString() + ".  Not enough positive examples.");
					return binaryValidation;
				}
					
				if (binaryValidation.run().get(0) < 0) {
					return null;
				}
				
				return binaryValidation;
			}
		});
		
		this.binaryValidations = threads.run(this.trainData.getDatumTools().getLabelIndicators(), this.maxThreads);
		this.aggregateConfusionMatrix = new ConfusionMatrix<T, Boolean>(binaryLabels);
		this.evaluationValues = new ArrayList<Double>();
		List<Double> evaluationCounts = new ArrayList<Double>();
		this.bestPositionCounts = new HashMap<GridSearch<T, Boolean>.GridPosition, Integer>();
		for (int i = 0; i < this.evaluations.size(); i++) {
			this.evaluationValues.add(0.0);
			evaluationCounts.add(0.0);
		}
		
		for (int i = 0; i < this.binaryValidations.size(); i++) {
			ValidationGST<T, Boolean> validation = this.binaryValidations.get(i);
			if (this.binaryValidations.get(i).trainData.getDataSizeForLabel(true)/(double)trainData.size() <= 0.0001
					|| this.binaryValidations.get(i).devData.getDataSizeForLabel(true) == 0
					|| (this.binaryValidations.get(i).testData != null && this.binaryValidations.get(i).testData.getDataSizeForLabel(true) == 0)) {
				output.resultsWriteln("Ignored " + this.trainData.getDatumTools().getLabelIndicators().get(i).toString() + " (lacking positive examples)");
				continue;
			}
			
			
			if (validation == null) {
				output.debugWriteln("ERROR: Validation thread failed.");
				return null;
			}
			
			ConfusionMatrix<T, Boolean> confusions = validation.getConfusionMatrix();
			
			for (int j = 0; j < this.evaluations.size(); j++) {
				this.evaluationValues.set(j, this.evaluationValues.get(j) + validation.getEvaluationValues().get(j));
				evaluationCounts.set(j, evaluationCounts.get(j) + 1); // FIXME: This is no longer necessary
			}
			
			this.aggregateConfusionMatrix.add(confusions);
			
			if (!this.bestPositionCounts.containsKey(validation.bestGridPosition))
				this.bestPositionCounts.put(validation.bestGridPosition, 0);
			this.bestPositionCounts.put(validation.bestGridPosition, this.bestPositionCounts.get(validation.bestGridPosition) + 1);
		}
		
		for (int j = 0; j < this.evaluations.size(); j++) {
			this.evaluationValues.set(j, this.evaluationValues.get(j) / evaluationCounts.get(j));
		}
		
		List<SupervisedModel<T, Boolean>> binaryModels = new ArrayList<SupervisedModel<T, Boolean>>(this.binaryValidations.size());
		for (ValidationGST<T, Boolean> binaryValidation : this.binaryValidations) {
			binaryModels.add(binaryValidation.getModel());
			
		}
		
		this.learnedCompositeModel = new SupervisedModelCompositeBinary<T, D, L>(
				binaryModels, 
				this.datumTools.getLabelIndicators(), 
				this.binaryValidations.get(0).datumTools, 
				this.inverseLabelIndicator);
		
		return this.evaluationValues;
	}
	
	@Override
	public boolean outputModel() {
		return true;
	}
	
	@Override
	public boolean outputData() { 
		return true;
	}
	
	@Override
	public boolean outputResults() {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		DecimalFormat cleanDouble = new DecimalFormat("0.00000");
		
		output.resultsWrite("\nMeasures:\t");
		for (SupervisedModelEvaluation<D, L> evaluation : this.evaluations)
			output.resultsWrite(evaluation.toString(false) + "\t");
		
		for (ValidationGST<T, Boolean> validation : this.binaryValidations) {
			if (validation.getEvaluationValues() == null)
				continue;
			
			output.resultsWrite("\n" + validation.name + ":\t");
			
			for (Double value : validation.getEvaluationValues())
				output.resultsWrite(cleanDouble.format(value) + "\t");
		}
		
		output.resultsWrite("\nAverages:\t");
		for (Double value : this.evaluationValues) 
			output.resultsWrite(cleanDouble.format(value) + "\t");
		
		output.resultsWriteln("\n\nConfusion matrix:\n" + this.aggregateConfusionMatrix.toString());
		
		output.resultsWriteln("\nBest grid search position counts:");
		for (Entry<GridSearch<T, Boolean>.GridPosition, Integer> entry : this.bestPositionCounts.entrySet())
			output.resultsWriteln(entry.getKey().toString() + "\t" + entry.getValue());	
		
		output.resultsWriteln("\nTime:\n" + this.datumTools.getDataTools().getTimer().toString());
	
		return true;
	}
	
	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("inverseLabelIndicator"))
			return this.inverseLabelIndicator.toString();
		else 
			return super.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		if (parameter.equals("inverseLabelIndicator")) {
			this.inverseLabelIndicator = datumTools.getInverseLabelIndicator(parameterValue);
			return true;
		} else 
			return super.setParameterValue(parameter, parameterValue, datumTools);
	}
}
