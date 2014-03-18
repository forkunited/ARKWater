package ark.data.feature;

import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.util.OutputWriter;

public class FeaturizedDataSet<D extends Datum<L>, L> extends DataSet<D, L> {
	private String name;
	private int maxThreads;
	private OutputWriter output;
	
	private List<Feature<D, L>> features;
	
	/* FIXME: Map indices to feature names. Map datum ids to indices to feature values */
	
	public FeaturizedDataSet(String name, OutputWriter output) {
		this(name, 1, output);
	}
	
	public FeaturizedDataSet(String name, int maxThreads, OutputWriter output) {
		this(name, new ArrayList<Feature<D, L>>(), maxThreads, output);
	}
	
	public FeaturizedDataSet(String name, List<Feature<D, L>> features, int maxThreads, OutputWriter output) {
		super();
		this.name = name;
		this.features = features;
		this.maxThreads = maxThreads;
		this.output = output;
		
		/* FIXME: Initialize other stuff */
	}
	
	public String getName() {
		return this.name;
	}
	
	public int getMaxThreads() {
		return this.maxThreads;
	}
	
	public boolean addFeature(Feature<D, L> feature) {
		return this.features.add(feature);
	}
	
	public boolean removeFeature(Feature<D, L> feature) {
		return this.features.remove(feature);
	}
	
	public Feature<D, L> getFeature(int index) {
		return this.features.get(index);
	}
	
	public List<Feature<D, L>>[] getFeatures() {
		/* FIXME */
		return null;
	}
	
	public int getFeatureCount() {
		return this.features.size();
	}
	
	public String[] getFeatureNames() {
		/* FIXME */
		return null;
	}
	
	public double[] getFeatureValues(D datum) {
		/* FIXME */
		return null;
	}
}
