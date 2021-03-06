/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of ARKWater (https://github.com/forkunited/ARKWater)
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

package ark.util;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

/**
 * MathUtil contains various math related utility
 * functions
 * 
 * @author Bill McDowell
 *
 */
public class MathUtil {
	/**
	 * 
	 * @param random
	 * @param array
	 * @return a permutation of array.  The permutation is computed inline,
	 * so the input list will be modified and returned.
	 */
	public static <T> T[] randomPermutation(Random random, T[] array) {
		for (int i = 0; i < array.length; i++) {
			int j = random.nextInt(i+1);
			T temp = array[i];
			array[i] = array[j];
			array[j] = temp;
		}
		return array;
	}
	
	/**
	 * 
	 * @param random
	 * @param list
	 * @return a permutation of list.  The permutation is computed inline,
	 * so the input list will be modified and returned.
	 */
	public static <T> List<T> randomPermutation(Random random, List<T> list) {
		for (int i = 0; i < list.size(); i++) {
			int j = random.nextInt(i+1);
			T temp = list.get(i);
			list.set(i, list.get(j));
			list.set(j, temp);
		}
		return list;
	}
	
	/**
	 * 
	 * @param n
	 * @param k
	 * @param r
	 * @return random sample of size k from the set [n]
	 */
	public static int[] reservoirSample(int n, int k, Random r) {
		int[] reservoir = new int[k];
		for (int i = 1; i < k+1; i++)
			reservoir[i-1] = i;
		
		for (int i = k+1; i < n+1; i++) {
			int j = r.nextInt(i);
			if (j < k)
				reservoir[j] = i;
		}
		
		return reservoir;
	}
	
	/**
	 * 
	 * @param min
	 * @param max
	 * @param r
	 * @return an integer sampled uniformly from the inclusive range 
	 * min...max 
	 */
	public static int uniformSample(int min, int max, Random r) {
		return min + r.nextInt(max - min + 1);
	}
	
	/**
	 * @param dist
	 * @return entropy of dist
	 */
	public static <T> double computeEntropy(Map<T, Double> dist) {
		double entropy = 0.0;
		for (Entry<T, Double> entry : dist.entrySet()) {
			double p = entry.getValue();
			entropy -= p * Math.log(p)/Math.log(2.0);
		}
		return entropy;
	}
	
	/**
	 * @param dist
	 * @return sum over components of dist
	 */
	public static <T> double computeSum(Map<T, Double> dist) {
		double sum = 0.0;
		for (Entry<T, Double> entry : dist.entrySet()) {
			sum += entry.getValue();
		}
		return sum;
	}
	
	/**
	 * @param dist
	 * @param norm
	 * @return dist normalized by norm
	 */
	public static <T> Map<T, Double> normalize(Map<T, Double> dist, double norm) {
		for (Entry<T, Double> entry : dist.entrySet()) {
			entry.setValue(entry.getValue() / norm);
		}
		return dist;
	}
}
