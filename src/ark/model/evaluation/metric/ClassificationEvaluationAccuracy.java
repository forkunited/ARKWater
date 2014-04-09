package ark.model.evaluation.metric;

import java.util.Collection;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.util.Pair;

public class ClassificationEvaluationAccuracy<D extends Datum<L>, L> extends ClassificationEvaluation<D, L> {

	@Override
	public double compute(Collection<Pair<L, L>> actualAndPredicted) {
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
	protected ClassificationEvaluation<D, L> makeInstance() {
		return new ClassificationEvaluationAccuracy<D, L>();
	}

}
