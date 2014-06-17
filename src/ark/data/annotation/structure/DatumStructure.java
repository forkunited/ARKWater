package ark.data.annotation.structure;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;

public abstract class DatumStructure<D extends Datum<L>, L> implements Collection<D> {
	protected interface DatumStructureOptimizer<D extends Datum<L>, L> {
		Map<D, L> optimize(Map<D, Map<L, Double>> scoredDatumLabels, Map<D, L> fixedDatumLabels, Set<L> validLabels, LabelMapping<L> labelMapping);
		String getGenericName();
	}
	
	protected String id;
	protected Map<String, DatumStructureOptimizer<D, L>> datumStructureOptimizers;
	
	public DatumStructure(String id) {
		this.id = id;
		this.datumStructureOptimizers = new HashMap<String, DatumStructureOptimizer<D, L>>();
	}
	
	public String getId() {
		return this.id;
	}
	
	public Map<D, L> getDatumLabels(LabelMapping<L> labelMapping) {
		Map<D, L> datumLabels = new HashMap<D, L>();
		
		for (D datum : this) {
			if (labelMapping == null)
				datumLabels.put(datum, datum.getLabel());
			else
				datumLabels.put(datum, labelMapping.map(datum.getLabel()));
		}
		
		return datumLabels;
	}
	
	@Override
	public int hashCode() {
		// FIXME: Make better
		return this.id.hashCode();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public boolean equals(Object o) {
		DatumStructure<D, L> datumStructure = (DatumStructure<D, L>)o;
		return datumStructure.id.equals(this.id);
	}
	
	public Map<D, L> optimize(String optimizerName, Map<D, Map<L, Double>> scoredDatumLabels, Map<D, L> fixedDatumLabels, Set<L> validLabels, LabelMapping<L> labelMapping) {
		return this.datumStructureOptimizers.get(optimizerName).optimize(scoredDatumLabels, fixedDatumLabels, validLabels, labelMapping);
	}
	
	protected boolean addDatumStructureOptimizer(DatumStructureOptimizer<D, L> datumStructureOptimizer) {
		this.datumStructureOptimizers.put(datumStructureOptimizer.getGenericName(), datumStructureOptimizer);
		return true;
	}
	
	public abstract Map<String, Integer> constraintsHold(boolean useDisjunctiveConstraints);
}
