package ark.model.evaluation.metric;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.util.Pair;

public class ClassificationEvaluationF<D extends Datum<L>, L> extends ClassificationEvaluation<D, L> {

	private boolean weighted;
	private double Beta = 1.0;
	private String[] parameterNames = { "weighted", "Beta" };
	
	@Override
	public double compute(Collection<Pair<L, L>> actualAndPredicted) {
		Map<L, Double> weights = new HashMap<L, Double>();
		Map<L, Double> tps = new HashMap<L, Double>();
		Map<L, Double> fps = new HashMap<L, Double>();
		Map<L, Double> fns = new HashMap<L, Double>();
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			L predicted = pair.getSecond();
			if (!weights.containsKey(actual)) {
				weights.put(actual, 0.0); 
				tps.put(actual, 0.0);
				fps.put(actual, 0.0);
				fns.put(actual, 0.0);
			}
			
			if (!weights.containsKey(predicted)) {
				weights.put(predicted, 0.0);
				tps.put(predicted, 0.0);
				fps.put(predicted, 0.0);
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
				fps.put(predicted, tps.get(predicted) + 1.0);
				fns.put(actual, fns.get(actual) + 1.0);
			}
		}
		
		double F = 0.0;
		double Beta2 = this.Beta*this.Beta;
		for (Entry<L, Double> weightEntry : weights.entrySet()) {
			L label = weightEntry.getKey();
			double weight = weightEntry.getValue();
			double tp = tps.get(label);
			double fp = fps.get(label);
			double fn = fns.get(label);
			
			if (tp == 0.0 && fn == 0.0 && fp == 0.0)
				F += weight;
			else
				F += weight*(1.0+Beta2)*tp/((1.0+Beta2)*tp + Beta2*fn + fp);
		}
		
		return F;
	}

	@Override
	public String getGenericName() {
		return "F";
	}

	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("weighted"))
			return String.valueOf(this.weighted);
		else if (parameter.equals("Beta"))
			return String.valueOf(this.Beta);
		else
			return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("weighted"))
			this.weighted = Boolean.valueOf(parameterValue);
		else if (parameter.equals("Beta"))
			this.Beta = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	protected ClassificationEvaluation<D, L> makeInstance() {
		return new ClassificationEvaluationF<D, L>();
	}
}
