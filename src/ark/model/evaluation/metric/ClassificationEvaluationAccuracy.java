package ark.model.evaluation.metric;

import java.util.Collection;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.util.Pair;

public class ClassificationEvaluationAccuracy<D extends Datum<L>, L> extends ClassificationEvaluation<D, L> {
	public Datum.Tools.LabelMapping<L> labelMapping;
	private String[] parameterNames = { "labelMapping" };
	
	@Override
	public double compute(Collection<Pair<L, L>> actualAndPredicted) {
		double total = actualAndPredicted.size();
		double correct = 0;
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = this.labelMapping.map(pair.getFirst());
			L predicted = this.labelMapping.map(pair.getSecond());
			
			if (actual.equals(predicted))
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
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("labelMapping"))
			return (this.labelMapping == null) ? null : this.labelMapping.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("labelMapping"))
			this.labelMapping = ((parameterValue == null) ? datumTools.getLabelMapping("Identity") : datumTools.getLabelMapping(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	protected ClassificationEvaluation<D, L> makeInstance() {
		return new ClassificationEvaluationAccuracy<D, L>();
	}

}
