package ark.model.evaluation.metric;

import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCL;

/**
 * This assumes the provided model is a cost learning model (with a loss function).  This could be redesigned
 * to get rid of this bad assumption
 * 
 * @author Bill
 */
public class SupervisedModelEvaluationCLLoss<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {

	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		SupervisedModelCL<D, L> modelCl = (SupervisedModelCL<D, L>)model;
		return modelCl.computeLoss(data);
	}

	@Override
	public String getGenericName() {
		return "CLLoss";
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
		return new SupervisedModelEvaluationCLLoss<D, L>();
	}
}