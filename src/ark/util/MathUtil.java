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

package ark.util;

import java.util.List;
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
}
