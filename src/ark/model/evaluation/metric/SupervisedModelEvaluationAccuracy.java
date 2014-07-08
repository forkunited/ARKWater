package ark.model.evaluation.metric;

import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;

/**
 * SupervisedModelEvaluationAccuracy computes the micro-averaged accuracy
 * for a supervised classification model on a data set.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelEvaluationAccuracy<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {

	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = getMappedActualAndPredictedLabels(predictions);
		double total = actualAndPredicted.size();
		double correct = 0;
		for (Pair<L, L> pair : actualAndPredicted) {
			if (pair.getFirst().equals(pair.getSecond()))
				correct += 1.0;
		}
		
		return (total == 0) ? 1 : correct/total;
	}

	@Override
	public String getGenericName() {
		return "Accuracy";
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
		return new SupervisedModelEvaluationAccuracy<D, L>();
	}

}
