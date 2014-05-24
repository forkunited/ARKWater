package ark.data.annotation;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.TreeMap;

public class DataSet<D extends Datum<L>, L> implements Collection<D> {
	private enum DataFilter {
		All,
		OnlyLabeled,
		OnlyUnlabeled
	}
	
	public class DataIterator implements Iterator<D> {
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
	
	private Datum.Tools<D, L> datumTools;
	private Datum.Tools.LabelMapping<L> labelMapping;
	
	private Map<L, List<Integer>> labeledData;
	private List<Integer> unlabeledData;
	protected TreeMap<Integer, D> data;
	
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
	
	public DataSet(Datum.Tools<D, L> datumTools, Datum.Tools.LabelMapping<L> labelMapping) {
		// Used treemap to ensure same ordering when iterating over data across
		// multiple runs.  HashMap might not give consistent ordering if L.hashCode()
		// is based on the Object reference (for example if L is an enum)
		// (This isn't an issue right now since the labeledData map is no longer used
		// to iterate through the data)
		this.labeledData = new TreeMap<L, List<Integer>>(this.labelComparator);
		this.unlabeledData = new ArrayList<Integer>();
		
		// For iterating in order by ID
		this.data = new TreeMap<Integer, D>();
		
		this.labelMapping = labelMapping;
		this.datumTools = datumTools;
	}
	
	public boolean addAll(Collection<? extends D> data) {
		for (D datum : data)
			if (!add(datum))
				return false;
		return true;
	}
	
	public boolean add(D datum) {
		L label = (this.labelMapping == null) ? datum.getLabel() : this.labelMapping.map(datum.getLabel());
		if (label == null) {
			this.unlabeledData.add(datum.getId());
		} else {
			if (!this.labeledData.containsKey(label))
				this.labeledData.put(label, new ArrayList<Integer>());
			this.labeledData.get(label).add(datum.getId());
		}
		
		this.data.put(datum.getId(), datum);
		
		return true;
	}
	
	public D getDatumById(int id) {
		return this.data.get(id);
	}
	
	public List<D> getDataForLabel(L label) {
		if (this.labelMapping != null)
			label = this.labelMapping.map(label);
		List<D> labelData = new ArrayList<D>();
		if (!this.labeledData.containsKey(label))
			return labelData;
		List<Integer> datumIds = this.labeledData.get(label);
		for (Integer datumId : datumIds)
			labelData.add(this.data.get(datumId));
		return labelData;
	}
	
	public List<DataSet<D, L>> makePartition(double[] distribution, Random random) {
		List<Integer> dataPermutation = constructRandomDataPermutation(random);
		List<DataSet<D, L>> partition = new ArrayList<DataSet<D, L>>(distribution.length);
		
		int offset = 0;
		for (int i = 0; i < distribution.length; i++) {
			int partSize = (int)Math.floor(this.data.size()*distribution[i]);
			if (i == distribution.length - 1 && offset + partSize < this.data.size())
				partSize = this.data.size() - offset;
			
			DataSet<D, L> part = new DataSet<D, L>(this.datumTools, this.labelMapping);
			for (int j = offset; j < offset + partSize; j++) {
				part.add(this.data.get(dataPermutation.get(j)));
			}
			
			offset += partSize;
			partition.add(part);
		}
		
		return partition;
	} 
	
	public Datum.Tools<D, L> getDatumTools() {
		return this.datumTools;
	}

	public Datum.Tools.LabelMapping<L> getLabelMapping() {
		return this.labelMapping;
	}
	
	@Override
	public Iterator<D> iterator() {
		return iterator(DataFilter.All);
	}
	
	public Iterator<D> iterator(DataFilter dataFilter) {
		return new DataIterator(dataFilter, this.data);
	}

	@SuppressWarnings("unchecked")
	@Override
	public boolean contains(Object obj) {
		Datum<L> datum = (Datum<L>)obj;
		return this.data.containsKey(datum.getId());
	}

	@Override
	public boolean containsAll(Collection<?> collection) {
		for (Object datum : collection)
			if (!contains(datum))
				return false;
		return true;
	}

	@Override
	public boolean isEmpty() {
		return this.data.isEmpty();
	}

	@Override
	public int size() {
		return this.data.size();
	}

	@Override
	public Object[] toArray() {
		Object[] array = new Object[this.data.size()];
		int i = 0;
		for (Entry<Integer, D> entry : this.data.entrySet()) {
			array[i] = entry.getValue();
			i++;
		}

		return array;
	}

	@SuppressWarnings("unchecked")
	@Override
	public <T> T[] toArray(T[] array) {
		if (array.length < this.data.size())
			array = (T[])Array.newInstance(array.getClass().getComponentType(), this.data.size());
		
		int i = 0;
		for (Entry<Integer, D> entry : this.data.entrySet()) {
			array[i] = (T)entry.getValue();
			i++;
		}
		
		if (i < array.length)
			array[i] = null;

		return array;
	}
	
	@Override
	public void clear() {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public boolean remove(Object arg0) {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean removeAll(Collection<?> arg0) {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean retainAll(Collection<?> arg0) {
		throw new UnsupportedOperationException();
	}
	
	public List<Integer> constructRandomDataPermutation(Random random) {
		List<Integer> permutation = new ArrayList<Integer>(this.data.size());
		for (Integer dataKey : this.data.keySet())
			permutation.add(dataKey);
		
		for (int i = 0; i < permutation.size(); i++) {
			int j = random.nextInt(i+1);
			int temp = permutation.get(i);
			permutation.set(i, permutation.get(j));
			permutation.set(j, temp);
		}
		return permutation;
	}
}
