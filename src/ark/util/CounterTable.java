package ark.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

/**
 * CounterTable represents a histogram.  It allows incrementing 
 * and decrementing counts for each item, and transforming the histogram
 * into various data-structures.
 * 
 * @author Lingpeng Kong, Bill McDowell
 * 
 */
public class CounterTable<T>{
	public HashMap<T, Integer> counts;
	
	public CounterTable(){
		this.counts= new HashMap<T,Integer>();
	}
	
	public void incrementCount(T w){
		if(this.counts.containsKey(w)){
			this.counts.put(w, this.counts.get(w) + 1);
		}else{
			this.counts.put(w, 1);
		}
	}
	
	public void removeCountsLessThan(int minCount) {
		List<T> valuesToRemove = new ArrayList<T>();
		for (Entry<T, Integer> entry : this.counts.entrySet()) {
			if (entry.getValue() < minCount)
				valuesToRemove.add(entry.getKey());
		}
		
		for (T valueToRemove : valuesToRemove)
			this.counts.remove(valueToRemove);
	}
	
	public Map<T, Integer> buildIndex() {
		HashMap<T, Integer> index = new HashMap<T, Integer>(this.counts.size());
		int i = 0;
		
		for (Entry<T, Integer> entry : this.counts.entrySet()) {
			index.put(entry.getKey(), i);
			i++;
		}
		
		return index;
	}
	
	public TreeMap<Integer, List<T>> getSortedCounts() {
		TreeMap<Integer, List<T>> sortedCounts = new TreeMap<Integer, List<T>>();
		
		for (Entry<T, Integer> entry : this.counts.entrySet()) {
			if (!sortedCounts.containsKey(entry.getValue()))
				sortedCounts.put(entry.getValue(), new ArrayList<T>());
			
			sortedCounts.get(entry.getValue()).add(entry.getKey());
		}
		
		return sortedCounts;
	}
	
	public Map<T, Integer> getCounts() {
		return this.counts;
	}
	
	public int getSize() {
		return this.counts.size();
	}
}