package ark.data;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import ark.util.FileUtil;

/**
 * Gazetteer represents a deserialized dictionary of strings
 * mapped to IDs.  The file in which the gazetteer is stored
 * should contain lines of the form:
 * 
 * [ID]	[string_1]	[string_2]	...	[string_n]
 * 
 * Each ID should only occur on a single line, but a string
 * can occur across multiple lines, to be mapped to multiple 
 * IDs.  The strings are cleaned by a specified clean function
 * as they are loaded into memory.  
 * 
 * @authors Lingpeng Kong, Bill McDowell
 *
 */
public class Gazetteer {
	private HashMap<String, List<String>> gazetteer;
	private String name;
	private DataTools.StringTransform cleanFn;
	
	public Gazetteer(String name, String sourceFilePath, DataTools.StringTransform cleanFn) {
		this.cleanFn = cleanFn;
		this.gazetteer = new HashMap<String, List<String>>();
		this.name = name;
		
		try {
			BufferedReader br = FileUtil.getFileReader(sourceFilePath);
			String line = null;
			while ((line = br.readLine()) != null) {
				String[] lineValues = line.trim().split("\\t");
				if (lineValues.length < 2) {
					continue;
				}
				
				String id = lineValues[0];
				for (int i = 1; i < lineValues.length; i++) {
					String cleanValue = cleanString(lineValues[i]);
					if (cleanValue.length() == 0)
						continue;
					if (!this.gazetteer.containsKey(cleanValue))
						this.gazetteer.put(cleanValue, new ArrayList<String>(2));
					if (!this.gazetteer.get(cleanValue).contains(id))
						this.gazetteer.get(cleanValue).add(id);
				}
			}
			
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public String getName() {
		return this.name;
	}
	
	private String cleanString(String str) {		
		return this.cleanFn.transform(str);
	}
	
	public boolean contains(String str) {
		return this.gazetteer.containsKey(cleanString(str));
	}
	
	public List<String> getIds(String str) {
		String cleanStr = cleanString(str);
		if (this.gazetteer.containsKey(cleanStr))
			return this.gazetteer.get(cleanStr);
		else
			return null;
	}
	
	public double min(String str, DataTools.StringPairMeasure fn) {
		double min = Double.POSITIVE_INFINITY;
		String cleanStr = cleanString(str);
		for (String gStr : this.gazetteer.keySet()) {
			double curMin = fn.compute(cleanStr, gStr);
			min = (curMin < min) ? curMin : min;
		}
		return min;
	}
	
	public double max(String str, DataTools.StringPairMeasure fn) {
		double max = Double.NEGATIVE_INFINITY;
		String cleanStr = cleanString(str);
		for (String gStr : this.gazetteer.keySet()) {
			double curMax = fn.compute(cleanStr, gStr);
			max = (curMax > max) ? curMax : max;
		}
		return max;
	}
	
	public String removeTerms(String str) {
		String[] strTokens = str.split("\\s+");
		StringBuilder termsRemoved = new StringBuilder();
		for (int i = 0; i < strTokens.length; i++) {
			if (!contains(strTokens[i]))
				termsRemoved.append(strTokens[i]).append(" ");
		}
		
		return termsRemoved.toString().trim();
	}
	
	public Set<String> getValues() {
		return this.gazetteer.keySet();
	}
}