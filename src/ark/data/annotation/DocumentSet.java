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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ark.util.MathUtil;

/**
 * FIXME Finish this class when re-factoring. 
 * 
 * DocumentSet represents a collection of text documents with NLP
 * annotations.  This class is currently not used, and there are 
 * domain-specific versions of it in the projects that use ARKWater.
 * It is only partially implemented.
 * 
 * @author Bill McDowell
 *
 * @param Document type
 */
public class DocumentSet<D extends Document> implements Collection<D> {
	private String name;
	private Map<String, D> documents;
	
	public DocumentSet(String name) {
		this.name = name;
		this.documents = new HashMap<String, D>();
	}
	
	public String getName() {
		return this.name;
	}
	
	public List<DocumentSet<D>> makePartition(int parts, Random random) {
		double[] distribution = new double[parts];
		String[] names = new String[distribution.length];
		for (int i = 0; i < distribution.length; i++) {
			names[i] = String.valueOf(i);
			distribution[i] = 1.0/parts;
		}
	
		return makePartition(distribution, names, random);
	}
	
	public List<DocumentSet<D>> makePartition(double[] distribution, Random random) {
		String[] names = new String[distribution.length];
		for (int i = 0; i < names.length; i++)
			names[i] = String.valueOf(i);
	
		return makePartition(distribution, names, random);
	}
	
	public List<DocumentSet<D>> makePartition(double[] distribution, String[] names, Random random) {
		List<D> documentList = new ArrayList<D>();
		documentList.addAll(this.documents.values());
		List<Integer> documentPermutation = new ArrayList<Integer>();
		for (int i = 0; i < documentList.size(); i++)
			documentPermutation.add(i);
		
		documentPermutation = MathUtil.randomPermutation(random, documentPermutation);
		List<DocumentSet<D>> partition = new ArrayList<DocumentSet<D>>(distribution.length);
		
		int offset = 0;
		for (int i = 0; i < distribution.length; i++) {
			int partSize = (int)Math.floor(documentList.size()*distribution[i]);
			if (i == distribution.length - 1 && offset + partSize < documentList.size())
				partSize = documentList.size() - offset;
			
			DocumentSet<D> part = new DocumentSet<D>(names[i]);
			for (int j = offset; j < offset + partSize; j++) {
				part.add(documentList.get(documentPermutation.get(j)));
			}
			
			offset += partSize;
			partition.add(part);
		}
		
		return partition;
	} 
	
	public boolean saveToJSONDirectory(String directoryPath) {
		for (D document : this.documents.values()) {
			document.saveToJSONFile(new File(directoryPath, document.getName() + ".json").getAbsolutePath());
		}
		
		return true;
	}
	
	
	@SuppressWarnings("unchecked")
	public static <D extends Document> DocumentSet<D> loadFromJSONDirectory(String name, String directoryPath, D genericDocument) {
		File directory = new File(directoryPath);
		DocumentSet<D> documentSet = new DocumentSet<D>(name);
		try {
			if (!directory.exists() || !directory.isDirectory())
				throw new IllegalArgumentException("Invalid directory: " + directory.getAbsolutePath());
			
			List<File> files = new ArrayList<File>();
			files.addAll(Arrays.asList(directory.listFiles()));
			int numTopLevelFiles = files.size();
			for (int i = 0; i < numTopLevelFiles; i++)
				if (files.get(i).isDirectory())
					files.addAll(Arrays.asList(files.get(i).listFiles()));
			
			List<File> tempFiles = new ArrayList<File>();
			for (File file : files) {
				if (!file.isDirectory() && file.getName().endsWith(".json")) {
					tempFiles.add(file);
				}
			}
			
			Collections.sort(tempFiles, new Comparator<File>() { // Ensure determinism
			    public int compare(File o1, File o2) {
			        return o1.getAbsolutePath().compareTo(o2.getAbsolutePath());
			    }
			});
			
			for (File file : tempFiles) {
				documentSet.add((D)genericDocument.makeInstanceFromJSONFile(file.getAbsolutePath()));
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}	
		
		return documentSet;
	}
	
	////////////////////
	
	@Override
	public boolean add(D e) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean addAll(Collection<? extends D> c) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void clear() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean contains(Object o) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean containsAll(Collection<?> c) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isEmpty() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Iterator<D> iterator() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean remove(Object o) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean removeAll(Collection<?> c) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean retainAll(Collection<?> c) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Object[] toArray() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public <T> T[] toArray(T[] a) {
		// TODO Auto-generated method stub
		return null;
	}

}
