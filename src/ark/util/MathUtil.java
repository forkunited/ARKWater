package ark.util;

import java.util.List;
import java.util.Random;

public class MathUtil {
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
