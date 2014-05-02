package ark.data.annotation.structure;

import java.util.Collection;
import ark.data.annotation.Datum;

public abstract class DatumStructure<D extends Datum<L>, L> implements Collection<D> {
	protected int id;
	
	@Override
	public int hashCode() {
		// FIXME: Make better
		return this.id;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public boolean equals(Object o) {
		DatumStructure<D, L> datumStructure = (DatumStructure<D, L>)o;
		return datumStructure.id == this.id;
	}
}
