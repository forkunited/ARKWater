package ark.data;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ark.wrapper.BrownClusterer;

import ark.util.OutputWriter;
import ark.util.StringUtil;
import ark.data.Gazetteer;

/**
 * 
 * DataTools loads various gazetteers and other data used in various 
 * models and experiments.  
 * 
 * Currently, for convenience, DataTools just loads everything into 
 * memory upon construction.  If memory conservation becomes particularly
 * important, then possibly this class should be rewritten to only keep 
 * things in memory when they are needed.
 * 
 * @author Bill McDowell
 *
 */
public class DataTools {
	public interface StringTransform {
		String transform(String str);
		// Return constant name for this transformation (used for deserializing features)
		String toString(); 
	}
	
	public interface StringPairMeasure {
		double compute(String str1, String str2);
	}
	
	public interface StringCollectionTransform {
		Collection<String> transform(String str);
		String toString();
	}
	
	public class Path {
		private String name;
		private String value;
		
		public Path(String name, String value) {
			this.name = name;
			this.value = value;
		}
		
		public String getValue() {
			return this.value;
		}
		
		public String getName() {
			return this.name;
		}
	}
	
	protected Map<String, Gazetteer> gazetteers;
	protected Map<String, DataTools.StringTransform> cleanFns;
	protected Map<String, DataTools.StringCollectionTransform> collectionFns;
	protected Map<String, BrownClusterer> brownClusterers;
	protected Map<String, Path> paths;
	
	protected OutputWriter outputWriter;
	
	public DataTools(OutputWriter outputWriter) {
		this.gazetteers = new HashMap<String, Gazetteer>();
		this.cleanFns = new HashMap<String, DataTools.StringTransform>();
		this.collectionFns = new HashMap<String, DataTools.StringCollectionTransform>();
		this.brownClusterers = new HashMap<String, BrownClusterer>();
		this.paths = new HashMap<String, Path>();
		
		this.outputWriter = outputWriter;
		
		this.cleanFns.put("DefaultCleanFn", new DataTools.StringTransform() {
			public String toString() {
				return "DefaultCleanFn";
			}
			
			public String transform(String str) {
				return StringUtil.clean(str);
			}
		});
		
		this.collectionFns.put("Prefixes", new DataTools.StringCollectionTransform() {
			public String toString() {
				return "Prefixes";
			}
			
			public Collection<String> transform(String str) {
				return StringUtil.prefixes(str);
			}
		});
		
		this.collectionFns.put("None", null);
		this.brownClusterers.put("None", null);
	}
	
	public Gazetteer getGazetteer(String name) {
		return this.gazetteers.get(name);
	}
	
	public DataTools.StringTransform getCleanFn(String name) {
		return this.cleanFns.get(name);
	}
	
	public DataTools.StringCollectionTransform getCollectionFn(String name) {
		return this.collectionFns.get(name);
	}
	
	public BrownClusterer getBrownClusterer(String name) {
		return this.brownClusterers.get(name);
	}
	
	public Path getPath(String name) {
		return this.paths.get(name);
	}
	
	public OutputWriter getOutputWriter() {
		return this.outputWriter;
	}
	
	public boolean addGazetteer(Gazetteer gazetteer) {
		this.gazetteers.put(gazetteer.getName(), gazetteer);
		return true;
	}
	
	public boolean addCleanFn(DataTools.StringTransform cleanFn) {
		this.cleanFns.put(cleanFn.toString(), cleanFn);
		return true;
	}
	
	public boolean addStopWordsCleanFn(final Gazetteer stopWords) {
		this.cleanFns.put("StopWordsCleanFn_" + stopWords.getName(), 
			new DataTools.StringTransform() {
				public String toString() {
					return "StopWordsCleanFn_" + stopWords.getName();
				}
				
				public String transform(String str) {
					str = StringUtil.clean(str);
					String stoppedStr = stopWords.removeTerms(str);
					if (stoppedStr.length() > 0)
						return stoppedStr;
					else 
						return str;
				}
			}
		);
		return true;
	}
	
	public boolean addCollectionFn(DataTools.StringCollectionTransform collectionFn) {
		this.collectionFns.put(collectionFn.toString(), collectionFn);
		return true;
	}
	
	public boolean addBrownClusterer(BrownClusterer brownClusterer) {
		this.brownClusterers.put(brownClusterer.toString(), brownClusterer);
		return true;
	}
	
	public boolean addPath(String name, Path path) {
		this.paths.put(name, path);
		return true;
	}
}
