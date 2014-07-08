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
