/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of theMess (https://github.com/forkunited/theMess)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy 
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations 
 * under the License.
 */

package ark.data.annotation;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.json.JSONException;
import org.json.JSONObject;

import ark.data.Context;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.util.MathUtil;
import ark.util.Pair;
import ark.util.ThreadMapper;

/**
 * DataSet represents a collection of labeled and/or unlabeled 'datums'
 * to be used to train and evaluate models.  
 * 
 * @author Bill McDowell
 *
 * @param <D> Datum type
 * @param <L> Datum label type
 */
public class DataSet<D extends Datum<L>, L> implements Collection<D> {
	public enum DataFilter {
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
	
	protected Map<L, Set<Integer>> labeledData;
	protected Set<Integer> unlabeledData;
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
		clear();
		
		this.labelMapping = labelMapping;
		this.datumTools = datumTools;
	}
	
	public boolean setDatumLabel(D datum, L label) {
		if (!this.contains(datum))
			return false;
		if (datum.getLabel() == null) {
			this.unlabeledData.remove(datum.getId());
			this.data.remove(datum.getId());	
			datum.setLabel(label);
			add(datum);
		} else {
			this.labeledData.get(datum.getLabel()).remove(datum.getId());
			this.data.remove(datum.getId());			
			datum.setLabel(label);
			add(datum);
		}
		
		return true;
	}
	
	public boolean addAll(Collection<? extends D> data) {
		for (D datum : data)
			if (!add(datum))
				return false;
		return true;
	}
	
	public boolean add(D datum) {
		if (contains(datum))
			return true;
		L label = (this.labelMapping == null) ? datum.getLabel() : this.labelMapping.map(datum.getLabel());
		if (label == null) {
			this.unlabeledData.add(datum.getId());
		} else {
			if (!this.labeledData.containsKey(label))
				this.labeledData.put(label, new HashSet<Integer>());
			this.labeledData.get(label).add(datum.getId());
		}
		
		this.data.put(datum.getId(), datum);
		
		return true;
	}
	
	public D getDatumById(int id) {
		return this.data.get(id);
	}
	
	public int getDataSizeForLabel(L label) {
		if (this.labelMapping != null)
			label = this.labelMapping.map(label);
		if (!this.labeledData.containsKey(label))
			return 0;
		else
			return this.labeledData.get(label).size();
	}
	
	public List<D> getDataForLabel(L label) {
		if (this.labelMapping != null)
			label = this.labelMapping.map(label);
		List<D> labelData = new ArrayList<D>();
		if (!this.labeledData.containsKey(label))
			return labelData;
		Set<Integer> datumIds = this.labeledData.get(label);
		for (Integer datumId : datumIds)
			labelData.add(this.data.get(datumId));
		return labelData;
	}
	
	public List<DataSet<D, L>> makePartition(int parts, Random random) {
		double[] distribution = new double[parts];
		for (int i = 0; i < distribution.length; i++)
			distribution[i] = 1.0/parts;
	
		return makePartition(distribution, random);
	}
	
	/**
	 * 
	 * @param distribution
	 * @param random
	 * @return a random partition of the dataset with whose sets have sizes given
	 * by the distribution
	 * 
	 */
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

	public <C> List<DataSet<D, L>> makePartition(double[] distribution, Datum.Tools.Clusterer<D, L, C> clusterer, Random random) {
		Map<C, List<D>> clusters = cluster(clusterer);
		List<C> clusterList = MathUtil.randomPermutation(random, new ArrayList<C>(clusters.keySet()));
		List<DataSet<D, L>> partition = new ArrayList<DataSet<D, L>>(distribution.length);
		
		int offset = 0;
		for (int i = 0; i < distribution.length; i++) {
			int partSize = (int)Math.floor(clusterList.size()*distribution[i]);
			if (i == distribution.length - 1 && offset + partSize < clusterList.size())
				partSize = clusterList.size() - offset;
			
			DataSet<D, L> part = new DataSet<D, L>(this.datumTools, this.labelMapping);
			for (int j = offset; j < offset + partSize; j++) {
				part.addAll(clusters.get(clusterList.get(j)));
			}
			
			offset += partSize;
			partition.add(part);
		}
		
		return partition;
	}
	
	public <C> Map<C, List<D>> cluster(Datum.Tools.Clusterer<D, L, C> clusterer) {
		Map<C, List<D>> clusters = new HashMap<C, List<D>>();
		
		for (D datum : this) {
			C cluster = clusterer.getCluster(datum);
			if (!clusters.containsKey(cluster))
				clusters.put(cluster, new ArrayList<D>());
			
			clusters.get(cluster).add(datum);
		}
		
		return clusters;
	}
	
	public DataSet<D, L> getSubset(DataFilter dataFilter) {
		DataSet<D, L> subset = new DataSet<D, L>(this.datumTools, this.labelMapping);
		
		Iterator<D> iterator = iterator(dataFilter);
		
		while (iterator.hasNext()) {
			subset.add(iterator.next());
		}
		
		return subset;
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
		// Used treemap to ensure same ordering when iterating over data across
		// multiple runs.  HashMap might not give consistent ordering if L.hashCode()
		// is based on the Object reference (for example if L is an enum)
		// (This isn't an issue right now since the labeledData map is no longer used
		// to iterate through the data)
		this.labeledData = new TreeMap<L, Set<Integer>>(this.labelComparator);
		this.unlabeledData = new HashSet<Integer>();
		
		// For iterating in order by ID
		this.data = new TreeMap<Integer, D>();
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
		
		return MathUtil.randomPermutation(random, permutation);
	}
	
	public Pair<L, Integer> computeMajorityLabel() {
		L maxLabel = null;
		int maxLabelCount = 0;
		for (Entry<L, Set<Integer>> entry : this.labeledData.entrySet()) {
			if (entry.getValue().size() > maxLabelCount) {
				maxLabel = entry.getKey();
				maxLabelCount = entry.getValue().size();
			}
		}
		
		return new Pair<L, Integer>(maxLabel, maxLabelCount);
	}
	
	public <T> List<T> map(final ThreadMapper.Fn<D, T> fn, int maxThreads) {
		List<DataSet<D, L>> threadDataPartition  = this.makePartition(maxThreads, this.datumTools.getDataTools().getGlobalRandom());
		
		ThreadMapper<DataSet<D, L>, List<T>> threadMapper 
		= new ThreadMapper<DataSet<D, L>, List<T>>(new ThreadMapper.Fn<DataSet<D, L>, List<T>>() {
			@Override
			public List<T> apply(DataSet<D, L> dataSet) {
				List<T> results = new ArrayList<T>(dataSet.size());
				for (D datum : dataSet) {
					T result = fn.apply(datum);
					results.add(result);
				}
				
				return results;
			}
		});
	
		List<List<T>> threadResults = threadMapper.run(threadDataPartition, maxThreads);
		List<T> results = new ArrayList<T>(size());
		for (List<T> threadResult : threadResults)
			results.addAll(threadResult);
		
		return results;
	}
	
	public boolean deserialize(BufferedReader reader, int idOffset) throws IOException {		
		try {
			String line = null;
			int id = idOffset;
			while ((line = reader.readLine()) != null) {
				JSONObject json = new JSONObject(line);
				D datum = this.datumTools.datumFromJSON(json);
				
				datum.id = id;
				id++;
				
				if (!add(datum))
					return false;
			}
			reader.close();
			return true;
		} catch (JSONException e) {
			e.printStackTrace();
			return false;
		}
	}
	
	public boolean serialize(Writer writer) throws IOException {
		for (D datum : this) {
			JSONObject json = this.datumTools.datumToJSON(datum);
			writer.write(json.toString());
			writer.write("\n");
		}
		writer.close();
		
		return true;
	}
	
	public <T extends Datum<Boolean>> DataSet<T, Boolean> makeBinary(Context<T, Boolean> context) {
		return makeBinary(null, context);
	}
	
	public <T extends Datum<Boolean>> DataSet<T, Boolean> makeBinary(LabelIndicator<L> labelIndicator, Context<T, Boolean> context) {
		DataSet<T, Boolean> dataSet = new DataSet<T, Boolean>(context.getDatumTools(), null);
		
		for (D datum : this) {
			dataSet.add(this.datumTools.<T>makeBinaryDatum(datum, labelIndicator));
		}
		
		return dataSet;
	}
}
