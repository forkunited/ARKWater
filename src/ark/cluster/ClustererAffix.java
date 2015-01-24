package ark.cluster;

import java.util.ArrayList;
import java.util.List;

public class ClustererAffix extends Clusterer<String> {
	private String name;
	private int maxAffixLength;
	
	public ClustererAffix(String name, int maxAffixLength) {
		this.name = name;
		this.maxAffixLength = maxAffixLength;
	}
	
	@Override
	public String getName() {
		return this.name;
	}
	
	@Override
	public List<String> getClusters(String str) {
		List<String> affixes = new ArrayList<String>();
		
		for (int i = 1; i <= this.maxAffixLength; i++) {
			if (i >= str.length())
				continue;
			
			String prefix = str.substring(0, i);
			String suffix = str.substring(str.length()-i, str.length());
			
			affixes.add("p_" + prefix);
			affixes.add("s_" + suffix);
		}
		
		return affixes;
	}
}
