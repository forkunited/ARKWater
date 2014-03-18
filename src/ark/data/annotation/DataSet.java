package ark.data.annotation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class DataSet<D extends Datum<L>, L> {
	private Map<L, List<D>> labeledData;
	private List<D> unlabeledData;
	private Comparator<L> labelComparator = new Comparator<L>() {
	      @Override
          public int compare(L o1, L o2) {
	    	  if (o1 == null && o2 == null)
	    		  return 0;
	    	  else if (o1 == null)
	    		  return -1;
	    	  else if (o2 == null)
	    		  return 1;
	    	  else 
	    		  return o1.toString().compareTo(o2.toString());
          }
	};
	
	public DataSet() {
		// Used treemap to ensure same ordering when iterating over data across
		// multiple runs.  Possibly not same ordering because CorpRelLabel.hashCode()
		// is from Object (based on reference).  This is because it's an enum.
		this.labeledData = new TreeMap<L, List<D>>(this.labelComparator);
		this.unlabeledData = new ArrayList<D>();
	}
	
	public boolean addData(List<D> data) {
		for (D datum : data)
			if (!addDatum(datum))
				return false;
		return true;
	}
	
	public boolean addDatum(D datum) {
		L label = datum.getLabel();
		if (label == null) {
			this.unlabeledData.add(datum);
			return true;
		} else {
			if (!this.labeledData.containsKey(label))
				this.labeledData.put(label, new ArrayList<D>());
			this.labeledData.get(label).add(datum);
		}
		
		return true;
	}
	
	public List<D> getLabeledData() {
		List<D> labeledData = new ArrayList<D>();
		for (List<D> data : this.labeledData.values())
			labeledData.addAll(data);
		return labeledData;
	}
	
	public List<D> getUnlabeledData() {
		return this.unlabeledData;
	}
	
	public List<D> getData() {
		List<D> data = getLabeledData();
		data.addAll(getUnlabeledData());
		return data;
	}
	
	public List<D> getDataForLabel(L label) {
		if (!this.labeledData.containsKey(label))
			return new ArrayList<D>();
		return this.labeledData.get(label);
	}
}
