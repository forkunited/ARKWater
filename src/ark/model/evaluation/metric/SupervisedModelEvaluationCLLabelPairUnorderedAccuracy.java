package ark.model.evaluation.metric;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCL;
import ark.model.cost.FactoredCostLabelPairUnordered;
import ark.util.Pair;

/**
 * This assumes the provided model is a cost learning model with a cost factored by unordered label pair.  
 * This could be redesigned to get rid of this bad assumption
 * 
 * @author Bill
 */
public class SupervisedModelEvaluationCLLabelPairUnorderedAccuracy<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {

	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		SupervisedModelCL<D, L> modelCl = (SupervisedModelCL<D, L>)model;
		FactoredCostLabelPairUnordered<D, L> factoredCost = (FactoredCostLabelPairUnordered<D, L>)modelCl.getFactoredCost();
		List<Pair<L, L>> labelPairs = factoredCost.getUnorderedLabelPairs();
		double[] labelPairWeights = modelCl.getCostWeights();
		final List<Pair<L, L>> collapsedLabelPairs = new ArrayList<Pair<L, L>>();
		for (int i = 0; i < labelPairWeights.length; i++) {
			if (labelPairWeights[i] == 0)
				collapsedLabelPairs.add(labelPairs.get(i));
		}
		
		LabelMapping<L> learnedCostMapping = new LabelMapping<L>() {
			@Override
			public String toString() {
				return "LearnedCostLabels";
			}

			@Override
			public L map(L label) {
				if (labelMapping != null)
					label = labelMapping.map(label);
				
				for (Pair<L, L> collapsedLabelPair : collapsedLabelPairs)
					if (collapsedLabelPair.getFirst().equals(label))
						return collapsedLabelPair.getSecond();
				return label;
			}
		};
		
		SupervisedModelEvaluation<D, L> accuracy = data.getDatumTools().makeEvaluationInstance("Accuracy");
		accuracy.labelMapping = learnedCostMapping;
		
		return accuracy.compute(modelCl, data, predictions);
	}

	@Override
	public String getGenericName() {
		return "CLLabelPairUnorderedAccuracy";
	}

	@Override
	protected String[] getParameterNames() {
		return new String[0];
	}

	@Override
	protected String getParameterValue(String parameter) {
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected SupervisedModelEvaluation<D, L> makeInstance() {
		return new SupervisedModelEvaluationCLLabelPairUnorderedAccuracy<D, L>();
	}
}