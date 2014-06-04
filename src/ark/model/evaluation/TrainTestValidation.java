package ark.model.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

public class TrainTestValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	private List<SupervisedModelEvaluation<D, L>> evaluations;
	private List<Double> results;
	private Pair<Long, Long> trainAndTestTime;
	
	public TrainTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> testData,
							  List<SupervisedModelEvaluation<D, L>> evaluations) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
		this.evaluations = evaluations;
		this.results = new ArrayList<Double>(this.evaluations.size());
	}
	
	public List<Double> run() {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		for (int i = 0; i < this.evaluations.size(); i++)
			this.results.add(-1.0);
		
		output.debugWriteln("Training model (" + this.name + ")");
		Long startTrain = System.currentTimeMillis();
		if (!this.model.train(this.trainData, this.testData, this.evaluations))
			return this.results;
		Long totalTrainTime = System.currentTimeMillis() - startTrain;
		
		output.debugWriteln("Classifying data (" + this.name + ")");
		
		Long startTest = System.currentTimeMillis();
		Map<D, L> classifiedData = this.model.classify(this.testData);
		if (classifiedData == null)
			return this.results;
		Long totalTestTime = System.currentTimeMillis() - startTest;
		this.trainAndTestTime = new Pair<Long, Long>(totalTrainTime, totalTestTime);
		
		output.debugWriteln("Computing model score (" + this.name + ")");
		
		for (int i = 0; i < this.evaluations.size(); i++)
			this.results.set(i, this.evaluations.get(i).evaluate(this.model, this.testData, classifiedData));
		
		this.confusionMatrix = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		this.confusionMatrix.addData(classifiedData);
		
		return this.results;
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public SupervisedModel<D, L> getModel() {
		return this.model;
	}
	
	public List<Double> getResults() {
		return this.results;
	}
	
	public Pair<Long, Long> getTrainAndTestTime(){
		return this.trainAndTestTime;
	}
}