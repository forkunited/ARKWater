package ark.model.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.ClassificationEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

public class TrainTestValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	private List<ClassificationEvaluation<D, L>> evaluations;
	
	public TrainTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> testData,
							  List<ClassificationEvaluation<D, L>> evaluations) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
		this.evaluations = evaluations;
	}
	
	public List<Double> run() {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		List<Double> results = new ArrayList<Double>(this.evaluations.size());
		for (int i = 0; i < this.evaluations.size(); i++)
			results.add(-1.0);
		
		output.debugWriteln("Training model (" + this.name + ")");
		if (!this.model.train(this.trainData))
			return results;
		
		output.debugWriteln("Classifying data (" + this.name + ")");
		
		Map<D, L> classifiedData =  this.model.classify(this.testData);
		if (classifiedData == null)
			return results;
		
		output.debugWriteln("Computing model score (" + this.name + ")");
		
		List<Pair<L, L>> actualAndPredicted = new ArrayList<Pair<L, L>>();
		for (Entry<D, L> classifiedDatum : classifiedData.entrySet()) {
			L actualLabel = this.model.mapValidLabel(classifiedDatum.getKey().getLabel());
			if (actualLabel == null)
				continue;
			actualAndPredicted.add(new Pair<L, L>(actualLabel, classifiedDatum.getValue()));
		}
		
		for (int i = 0; i < this.evaluations.size(); i++)
			results.set(i, this.evaluations.get(i).evaluate(actualAndPredicted));
		
		this.confusionMatrix = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		this.confusionMatrix.addData(classifiedData);
		
		return results;
	}
	
	public List<ClassificationEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public SupervisedModel<D, L> getModel() {
		return this.model;
	}
}