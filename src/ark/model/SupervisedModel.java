package ark.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.util.FileUtil;

public abstract class SupervisedModel<D extends Datum<L>, L> {
	protected String modelPath;
	protected List<L> validLabels;
	protected Map<String, Double> hyperParameters;
		
	public abstract boolean deserialize(String modelPath, DataTools dataTools, Datum<L>.AnnotationTools<D> annotationTools);
	public abstract boolean train(FeaturizedDataSet<D, L> data, String outputPath);
	public abstract Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data);
	public abstract SupervisedModel<D, L> clone();
	
	public List<L> getValidLabels() {
		return this.validLabels;
	}
	
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>();
		Map<D, Map<L, Double>> posterior = posterior(data);
	
		for (Entry<D, Map<L, Double>> datumPosterior : posterior.entrySet()) {
			Map<L, Double> p = datumPosterior.getValue();
			double max = Double.NEGATIVE_INFINITY;
			L argMax = null;
			for (Entry<L, Double> entry : p.entrySet()) {
				if (entry.getValue() > max) {
					max = entry.getValue();
					argMax = entry.getKey();
				}
			}
			classifiedData.put(datumPosterior.getKey(), argMax);
		}
	
		return classifiedData;
	}
		
	public void setHyperParameters(Map<String, Double> values) {
		for (Entry<String, Double> entry : values.entrySet()) {
			setHyperParameter(entry.getKey(), entry.getValue());
		}
	}
	
	public void setHyperParameter(String parameter, double value) {
		this.hyperParameters.put(parameter, value);
	}
	
	public double getHyperParameter(String parameter) {
		return this.hyperParameters.get(parameter);
	}
	
	public boolean hasHyperParameter(String parameter) {
		return this.hyperParameters.containsKey(parameter);
	}
	
	protected boolean serializeParameters() {
		if (this.modelPath == null)
			return false;
		
        try {
    		BufferedWriter w = new BufferedWriter(new FileWriter(this.modelPath + ".p"));  		
    		
    		for (L label : this.validLabels) {
    			w.write(label.toString() + "\t");
    		}
    		w.write("\n");
    		
    		for (Entry<String, Double> hyperParameter : this.hyperParameters.entrySet()) {
    			w.write(hyperParameter.getKey() + "\t" + hyperParameter.getValue() + "\n");
    		}
    		
            w.close();
            return true;
        } catch (IOException e) { e.printStackTrace(); return false; }
	}
	
	protected boolean deserializeParameters(Datum<L>.AnnotationTools<D> annotationTools) {
		if (this.modelPath == null)
			return false;
		
        try {
    		BufferedReader r = FileUtil.getFileReader(this.modelPath + ".p");		
    		
    		String validLabelsLine = r.readLine();
    		String[] validLabelStrs = validLabelsLine.trim().split("\t");
    		this.validLabels = new ArrayList<L>();
    		for (String validLabelStr : validLabelStrs) {
    			this.validLabels.add(annotationTools.labelFromString(validLabelStr));
    		}
    		
    		String hyperParameterLine = null;
    		this.hyperParameters = new HashMap<String, Double>();
    		while ((hyperParameterLine = r.readLine()) != null) {
    			String[] hyperParameterParts = hyperParameterLine.split("\t");
    			setHyperParameter(hyperParameterParts[0], Double.parseDouble(hyperParameterParts[1]));
    		}

    		r.close();
            return true;
        } catch (IOException e) { e.printStackTrace(); return false; }
	}
}
