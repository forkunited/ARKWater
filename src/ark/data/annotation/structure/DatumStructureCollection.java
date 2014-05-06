package ark.data.annotation.structure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;

public abstract class DatumStructureCollection<D extends Datum<L>, L> implements Iterable<DatumStructure<D, L>> {
	protected List<DatumStructure<D, L>> datumStructures;
	
	public abstract String getGenericName();
	public abstract DatumStructureCollection<D, L> makeInstance(DataSet<D, L> data);
	
	public DatumStructureCollection() {
		this.datumStructures = new ArrayList<DatumStructure<D, L>>();
	}

	public boolean isEmpty() {
		return this.datumStructures.isEmpty();
	}

	public Iterator<DatumStructure<D, L>> iterator() {
		return this.datumStructures.iterator();
	}

	public int size() {
		return this.datumStructures.size();
	}
	
	public DatumStructure<D, L> getDatumStructure(int index) {
		return this.datumStructures.get(index);
	}
}
