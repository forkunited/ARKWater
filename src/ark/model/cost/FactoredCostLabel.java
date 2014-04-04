package ark.model.cost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.model.SupervisedModel;

public class FactoredCostLabel<D extends Datum<L>, L> extends FactoredCost<D, L> {
	public enum FactorMode {
		Actual,
		Predicted
	}
	
	private String[] parameterNames = { "c", "factorMode" };
	private double c;
	private FactorMode factorMode;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	
	public FactoredCostLabel() {
		this.labels = new ArrayList<L>();
		this.factorMode = FactorMode.Actual;
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual))
			return vector;
		
		for (int i = 0; i < this.labels.size(); i++) {
			if ((this.factorMode.equals(FactorMode.Predicted) && this.labels.get(i).equals(prediction))
					|| (this.factorMode.equals(FactorMode.Actual) && this.labels.get(i).equals(actual))) {
				vector.put(i, this.c);
			}
		}
		
		return vector;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("c"))
			return String.valueOf(this.c);
		else if (parameter.equals("factorMode"))
			return this.factorMode.toString();
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("factorMode"))
			this.factorMode = FactorMode.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model) {
		this.model = model;
		this.labels.addAll(this.model.getValidLabels());
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "Label";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size();
	}

	@Override
	protected String getVocabularyTerm(int index) {
		return this.labels.get(index).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabel<D, L>();
	}

}
