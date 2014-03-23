package ark.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import net.sf.json.JSONObject;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.util.CommandRunner;
import ark.util.OutputWriter;

public class SupervisedModelCreg<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private String cmdPath;
	private String modelPath;
	private double l1;
	private double l2;
	private boolean warmRestart;
	private String[] hyperParameterNames = { "cmdPath", "modelPath", "l1", "l2", "warmRestart" };

	@Override
	public boolean train(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		String trainXPath = this.modelPath + ".train.x";
		String trainYPath = this.modelPath + ".train.y";
		
		output.debugWriteln("Creg outputting training data for (" + this.modelPath + ")");
		
		if (!outputXData(trainXPath, data, true))
			return false;
		if (!outputYData(trainYPath, data))
			return false;
		
		output.debugWriteln("Creg training model for (" + this.modelPath + ")");
		
		File outputFile = new File(this.modelPath);
		
		String trainCmd = this.cmdPath + 
						" -x " + trainXPath + 
						" -y " + trainYPath + 
						" --l1 " + this.l1 + 
						" --l2 " + this.l2 + 
						((this.warmRestart && outputFile.exists()) ? " --weights " + this.modelPath : "") +
						" --z " + this.modelPath;
		trainCmd = trainCmd.replace("\\", "/"); 
		if (!CommandRunner.run(trainCmd))
			return false;
		
		output.debugWriteln("Creg finished training model (" + this.modelPath + ")");
		
		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		String predictPPath = predict(data);
		if (predictPPath == null)
			return null;
		else 
			return loadPData(predictPPath, data, false);
	}
	
	private boolean outputXData(String outputPath, FeaturizedDataSet<D, L> data, boolean requireLabels) {
        try {
    		BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));
  
    		for (D datum : data) {
    			L label = mapValidLabel(datum.getLabel());
    			if (requireLabels && label == null)
    				continue;
    			
    			Map<Integer, Double> featureValues = data.getFeatureVocabularyValues(datum);
    			Map<Integer, String> featureNames = data.getFeatureVocabularyNamesForIndices(featureValues.keySet());
    			
    			StringBuilder datumStr = new StringBuilder();
    			datumStr = datumStr.append("id").append(datum.getId()).append("\t{");
    			for (Entry<Integer, Double> feature : featureValues.entrySet()) {
    				datumStr = datumStr.append("\"")
    								   .append(featureNames.get(feature.getKey()))
    								   .append("\": ")
    								   .append(feature.getValue())
    								   .append(", ");
    			}
    			
    			if (datumStr.length() > 0) {
    				datumStr = datumStr.delete(datumStr.length() - 2, datumStr.length());
    			}
    			
				datumStr = datumStr.append("}");
				
				writer.write(datumStr.toString());
				writer.write("\n");
    		}    		
    		
            writer.close();
            return true;
        } catch (IOException e) { e.printStackTrace(); return false; }
	}
	
	private boolean outputYData(String outputPath, FeaturizedDataSet<D, L> data) {
        try {
    		BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));

    		for (D datum : data) {
    			L label = mapValidLabel(datum.getLabel());
    			if (label == null)
    				continue;
    			
    			StringBuilder labelStr = new StringBuilder();
    			labelStr = labelStr.append("id")
    							   .append(datum.getId())
    							   .append("\t")
    							   .append(label.toString());
    			
				writer.write(labelStr.toString());
				writer.write("\n");
    		}    		
    		
            writer.close();
            return true;
        } catch (IOException e) { e.printStackTrace(); return false; }
	}

	private Map<D, Map<L, Double>> loadPData(String path, FeaturizedDataSet<D, L> data, boolean requireLabels) {
		Map<D, Map<L, Double>> pData = new HashMap<D, Map<L, Double>>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			
			for (D datum : data) {
				L label = mapValidLabel(datum.getLabel());
    			if (requireLabels && label == null)
    				continue;
				
				String line = br.readLine();
				if (line == null) {
					br.close();
					return null;
				}
					
				String[] lineParts = line.split("\t");
				if (lineParts.length < 3) {
					br.close();
					return null;
				}
				
				if (!lineParts[0].equals("id" + datum.getId())) {
					br.close();
					return null;
				}
				
				JSONObject jsonPosterior = JSONObject.fromObject(lineParts[2]);
				Map<L, Double> posterior = new HashMap<L, Double>();
				for (L validLabel : this.validLabels) {
					String labelStr = validLabel.toString();
					if (jsonPosterior.containsKey(labelStr))
						posterior.put(validLabel, jsonPosterior.getDouble(labelStr));
				}
				
				pData.put(datum, posterior);
			}
	        
	        br.close();
	    } catch (Exception e) {
	    	e.printStackTrace();
	    	return null;
	    }
		
		return pData;
	}
	
	private String predict(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		String predictXPath = this.modelPath + ".predict.x";
		String predictOutPath = this.modelPath + ".predict.y";
		
		output.debugWriteln("Creg outputting prediction data (" + this.modelPath + ")");
		
		if (!outputXData(predictXPath, data, false)) {
			output.debugWriteln("Error: Creg failed to output feature data (" + this.modelPath + ")");
			return null;
		}
		
		String predictCmd = this.cmdPath + " -w " + this.modelPath + " -W -D --tx " + predictXPath + " > " + predictOutPath;
		predictCmd = predictCmd.replace("\\", "/"); 
		if (!CommandRunner.run(predictCmd)) {
			output.debugWriteln("Error: Creg failed to run on output data (" + this.modelPath + ")");
			return null;
		}
		
		output.debugWriteln("Creg predicting data (" + this.modelPath + ")");
		
		return predictOutPath;
	}
	
	@Override
	protected boolean deserializeParameters(Reader reader, Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected boolean serializeParameters(Writer writer) {
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) {
		return true;
	}

	@Override
	public String getGenericName() {
		return "Creg";
	}
	
	@Override
	public String getHyperParameterValue(String parameter) {
		if (parameter.equals("cmdPath"))
			return this.cmdPath;
		else if (parameter.equals("modelPath"))
			return this.modelPath;
		else if (parameter.equals("l1"))
			return String.valueOf(this.l1);
		else if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("warmRestart"))
			return String.valueOf(this.warmRestart);
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("cmdPath"))
			this.cmdPath = datumTools.getDataTools().getPath(parameterValue);
		else if (parameter.equals("modelPath"))
			this.modelPath = datumTools.getDataTools().getPath(parameterValue);
		else if (parameter.equals("l1"))
			this.l1 = Double.valueOf(parameterValue);
		else if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("warmRestart"))
			this.warmRestart = Boolean.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	protected String[] getHyperParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelCreg<D, L>();
	}

	@Override
	protected boolean deserializeExtraInfo(Reader reader, Tools<D, L> datumTools) {
		return true;
	}

}
