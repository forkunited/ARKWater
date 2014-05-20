package ark.model.cost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

public class FactoredCostLabelPair<D extends Datum<L>, L> extends FactoredCost<D, L> {
	public enum Norm {
		NONE,
		LOGICAL,
		EXPECTED
	}
	
	
	private String[] parameterNames = { "c", "norm" };
	private double c;
	private Norm norm = Norm.NONE;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	private double[] norms;
	
	public FactoredCostLabelPair() {
		this.labels = new ArrayList<L>();
		this.norms = new double[0];
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual) || actual == null || prediction == null)
			return vector;
		
		int actualIndex = this.labels.indexOf(actual);
		int predictedIndex = this.labels.indexOf(prediction);
		int n = this.labels.size();
		vector.put(actualIndex*(n-1)+((predictedIndex > actualIndex) ? predictedIndex-1 : predictedIndex), this.c);
		
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
		else if (parameter.equals("norm"))
			return this.norm.toString();
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("norm"))
			this.norm = Norm.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data) {
		this.model = model;
		this.labels = new ArrayList<L>();
		this.labels.addAll(this.model.getValidLabels());
		
		int N = data.size();
		int vocabularySize = getVocabularySize();
		this.norms = new double[vocabularySize];
		Map<L, Integer> dist = new HashMap<L, Integer>();
		for (D datum : data) {
			L label = model.mapValidLabel(datum.getLabel());
			if (!dist.containsKey(label))
				dist.put(label, 0);
			dist.put(label, dist.get(label) + 1);
		}
		
		if (this.norm == Norm.EXPECTED) {	
			for (int i = 0; i < vocabularySize; i++) {
				int actualIndex = i / (this.labels.size() - 1);
				int rowPosition = i % (this.labels.size() - 1);
				int predictedIndex = rowPosition < actualIndex ? rowPosition : rowPosition + 1;
				
				L actualLabel = this.labels.get(actualIndex);
				L predictedLabel = this.labels.get(predictedIndex);
				double actualCount = dist.containsKey(actualLabel) ? dist.get(actualLabel) : 0;
				double predictedCount = dist.containsKey(predictedLabel) ? dist.get(predictedLabel) : 0;
				
				this.norms[i] = actualCount*predictedCount/N;
			}
		} else if (this.norm == Norm.LOGICAL) {
			for (int i = 0; i < vocabularySize; i++) {
				int actualIndex = i / (this.labels.size() - 1);
				L actualLabel = this.labels.get(actualIndex);
				double actualCount = dist.containsKey(actualLabel) ? dist.get(actualLabel) : 0;
				this.norms[i] = actualCount;
			}
		} else { 
			for (int i = 0; i < vocabularySize; i++) {
				this.norms[i] = N;
			}
		}
		
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "LabelPair";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size() * (this.labels.size() - 1);
	}

	@Override
	protected String getVocabularyTerm(int index) {
		int n = this.labels.size();
		int actualIndex = index / (n - 1);
		int rowPosition = index % (n - 1);
		int predictedIndex = rowPosition < actualIndex ? rowPosition : rowPosition + 1;
		
		return "A_" + this.labels.get(actualIndex).toString() + "P_" + this.labels.get(predictedIndex).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabelPair<D, L>();
	}

	@Override
	public Map<Integer, Double> computeKappas(Map<D, L> predictions) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public double[] getNorms() {
		return this.norms;
	}
}
