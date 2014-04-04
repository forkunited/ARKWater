package ark.model.cost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.model.SupervisedModel;

public class FactoredCostLabelPairUnordered<D extends Datum<L>, L> extends FactoredCost<D, L> {
	
	private String[] parameterNames = { "c" };
	private double c;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	
	public FactoredCostLabelPairUnordered() {
		this.labels = new ArrayList<L>();
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual) || actual == null || prediction == null)
			return vector;
		
		int actualIndex = this.labels.indexOf(actual);
		int predictedIndex = this.labels.indexOf(prediction);
		
		int rowIndex = (actualIndex < predictedIndex) ? actualIndex : predictedIndex;
		int columnIndex = (actualIndex < predictedIndex) ? predictedIndex : actualIndex;
		vector.put(rowIndex*(rowIndex-1)/2+columnIndex, this.c);
		
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
		this.labels.addAll(this.model.getValidLabels());
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "LabelPairUnordered";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size() * (this.labels.size() - 1)/2;
	}

	@Override
	protected String getVocabularyTerm(int index) {
		int rowIndex = (int)Math.floor(0.5*(Math.sqrt(8*index+1)+1));
		int columnIndex = index - rowIndex*(rowIndex-1)/2;
		return this.labels.get(rowIndex).toString() + "_" + this.labels.get(columnIndex).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabelPairUnordered<D, L>();
	}
}
