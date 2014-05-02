package ark.data.annotation.structure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;

public abstract class DatumStructureCollection<G extends DatumStructure<D, L>, D extends Datum<L>, L> {
	protected List<G> datumStructures;
	
	public abstract String getGenericName();
	public abstract DatumStructureCollection<G, D, L> makeInstance(DataSet<D, L> data);
	
	public DatumStructureCollection() {
		this.datumStructures = new ArrayList<G>();
	}

	public boolean isEmpty() {
		return this.datumStructures.isEmpty();
	}

	public Iterator<G> iterator() {
		return this.datumStructures.iterator();
	}

	public int size() {
		return this.datumStructures.size();
	}
	
	public G getDatumStructure(int index) {
		return this.datumStructures.get(index);
	}
}
