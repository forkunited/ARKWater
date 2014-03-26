package ark.model.evaluation.metric;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.util.Pair;

public class ClassificationEvaluationRecall<D extends Datum<L>, L> extends ClassificationEvaluation<D, L> {
	private boolean weighted;
	private String[] parameterNames = { "weighted" };
	
	@Override
	public double compute(Collection<Pair<L, L>> actualAndPredicted) {
		Map<L, Double> weights = new HashMap<L, Double>();
		Map<L, Double> tps = new HashMap<L, Double>();
		Map<L, Double> fns = new HashMap<L, Double>();
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			L predicted = pair.getSecond();
			if (!weights.containsKey(actual)) {
				weights.put(actual, 0.0); 
				tps.put(actual, 0.0);
				fns.put(actual, 0.0);
			}
			
			if (!weights.containsKey(predicted)) {
				weights.put(predicted, 0.0);
				tps.put(predicted, 0.0);
				fns.put(predicted, 0.0);
			}
			
			weights.put(actual, weights.get(actual) + 1.0);
		}
		
		for (Entry<L, Double> entry : weights.entrySet()) {
			if (this.weighted)
				entry.setValue(entry.getValue()/actualAndPredicted.size());
			else
				entry.setValue(1.0/weights.size());
		}
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			L predicted = pair.getSecond();
			
			if (actual.equals(predicted)) {
				tps.put(predicted, tps.get(predicted) + 1.0);
			} else {
				fns.put(actual, fns.get(actual) + 1.0);
			}
		}
		
		double recall = 0.0;
		for (Entry<L, Double> weightEntry : weights.entrySet()) {
			L label = weightEntry.getKey();
			double weight = weightEntry.getValue();
			double tp = tps.get(label);
			double fn = fns.get(label);
			
			if (tp == 0.0 && fn == 0.0)
				recall += weight;
			else
				recall += weight*tp/(tp + fn);
		}
		
		return recall;
	}

	@Override
	public String getGenericName() {
		return "Recall";
	}

	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("weighted"))
			return String.valueOf(this.weighted);
		else
			return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("weighted"))
			this.weighted = Boolean.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	protected ClassificationEvaluation<D, L> makeInstance() {
		return new ClassificationEvaluationRecall<D, L>();
	}

}
