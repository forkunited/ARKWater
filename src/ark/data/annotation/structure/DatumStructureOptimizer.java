package ark.data.annotation.structure;

import java.util.Map;

import ark.data.annotation.Datum;

public interface DatumStructureOptimizer<G extends DatumStructure<D, L>, D extends Datum<L>, L> {
	Map<D, L> optimize(G datumStructure, Map<D, Map<L, Double>> scoredDatumLabels, Map<D, L> fixedDatumLabels);
	String getGenericName();
}
