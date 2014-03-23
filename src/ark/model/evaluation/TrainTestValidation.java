package ark.model.evaluation;

import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.OutputWriter;

public class TrainTestValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	
	public TrainTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> testData) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
	}
	
	public double run() {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		output.debugWriteln("Training model (" + this.name + ")");
		if (!this.model.train(this.trainData))
			return -1.0;
		
		output.debugWriteln("Classifying data (" + this.name + ")");
		
		Map<D, L> classifiedData =  this.model.classify(this.testData);
		if (classifiedData == null)
			return -1.0;
		
		output.debugWriteln("Computing model score (" + this.name + ")");
		
		double correct = 0;
		double total = 0;
		for (Entry<D, L> classifiedDatum : classifiedData.entrySet()) {
			L actualLabel = this.model.mapValidLabel(classifiedDatum.getKey().getLabel());
			if (actualLabel == null)
				continue;
			correct += actualLabel.equals(classifiedDatum.getValue()) ? 1.0 : 0;
			total++;
		}
		
		this.confusionMatrix = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		this.confusionMatrix.addData(classifiedData);
		
		return correct/total;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
}