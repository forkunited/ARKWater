package ark.model.cost;

import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.model.SupervisedModel;

public class FactoredCostConstant<D extends Datum<L>, L> extends FactoredCost<D, L> {
	private String[] parameterNames = { "c" };
	private double c;
	private SupervisedModel<D, L> model;
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		if (!prediction.equals(this.model.mapValidLabel(datum.getLabel())))
			vector.put(0, this.c);
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
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model) {
		this.model = model;
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "Constant";
	}

	@Override
	public int getVocabularySize() {
		return 1;
	}

	@Override
	protected String getVocabularyTerm(int index) {
		return null;
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostConstant<D, L>();
	}
}
