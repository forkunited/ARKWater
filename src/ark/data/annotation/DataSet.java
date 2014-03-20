package ark.data.annotation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.TreeMap;

public class DataSet<D extends Datum<L>, L> implements Iterable<D> {
	private enum DataFilter {
		All,
		OnlyLabeled,
		OnlyUnlabeled
	}
	
	private class DataIterator implements Iterator<D> {
		private DataFilter filter;
		private Iterator<Entry<Integer, D>> iterator;
		private D next;
		
		public DataIterator(DataFilter filter, Map<Integer, D> data) {
			this.filter = filter;
			this.iterator = data.entrySet().iterator();
			iterate();
		}
		
		@Override
		public boolean hasNext() {
			return this.next != null;
		}

		@Override
		public D next() {
			if (this.next == null)
				throw new NoSuchElementException();
			
			D next = this.next;
			
			iterate();
			
			return next;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
		
		private void iterate() {
			do {
				if (this.iterator.hasNext())
					this.next = this.iterator.next().getValue();
				else
					this.next = null;
			} while (this.next != null && 
						(  (this.filter == DataFilter.OnlyLabeled && this.next.getLabel() == null)
						|| (this.filter == DataFilter.OnlyUnlabeled && this.next.getLabel() != null)));
		}
		
	}
	
	private Map<L, List<Integer>> labeledData;
	private List<Integer> unlabeledData;
	private TreeMap<Integer, D> data;
	
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
		// multiple runs.  HashMap might not give consistent ordering if L.hashCode()
		// is based on the Object reference (for example if L is an enum)
		// (This isn't an issue right now since the labeledData map is no longer used
		// to iterate through the data)
		this.labeledData = new TreeMap<L, List<Integer>>(this.labelComparator);
		this.unlabeledData = new ArrayList<Integer>();
		
		// For iterating in order by ID
		this.data = new TreeMap<Integer, D>();
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
			this.unlabeledData.add(datum.getId());
			return true;
		} else {
			if (!this.labeledData.containsKey(label))
				this.labeledData.put(label, new ArrayList<Integer>());
			this.labeledData.get(label).add(datum.getId());
		}
		
		this.data.put(datum.getId(), datum);
		
		return true;
	}
	
	public List<D> getDataForLabel(L label) {
		List<D> labelData = new ArrayList<D>();
		if (!this.labeledData.containsKey(label))
			return labelData;
		List<Integer> datumIds = this.labeledData.get(label);
		for (Integer datumId : datumIds)
			labelData.add(this.data.get(datumId));
		return labelData;
	}

	@Override
	public Iterator<D> iterator() {
		return iterator(DataFilter.All);
	}
	
	public Iterator<D> iterator(DataFilter dataFilter) {
		return new DataIterator(dataFilter, this.data);
	}
}
