package ark.data.feature;

import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;

public abstract class Feature<D extends Datum<L>, L> {
	public abstract void init(List<D> data);
	public abstract void init(String initStr);
	
	public abstract List<String> getNames();
	public abstract Map<Integer, Double> computeVector(D datum);
	public abstract Feature<D, L> clone();
	public abstract String toString(boolean withInit);
	
	public String toString() {
		return toString(false);
	}
}
