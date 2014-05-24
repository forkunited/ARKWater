package ark.experiment;

import java.io.BufferedReader;
import java.io.IOException;

import ark.data.annotation.Datum;
import ark.util.FileUtil;
import ark.util.SerializationUtil;

public abstract class Experiment<D extends Datum<L>, L> {
	protected String name;
	protected String inputPath;
	protected Datum.Tools<D, L> datumTools;
	protected int maxThreads;
	
	public Experiment(String name, String inputPath, Datum.Tools<D, L> datumTools) {
		this.name = name;
		this.inputPath = inputPath;
		this.datumTools = datumTools;
	}
	
	protected abstract boolean execute();
	protected abstract boolean deserializeNext(BufferedReader reader, String nextName) throws IOException;
	
	public boolean run() {
		try {
			return deserialize() && execute();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	protected boolean deserialize() throws IOException {
		BufferedReader reader = FileUtil.getFileReader(this.inputPath);
		String assignmentLeft = null;

		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("randomSeed"))
				this.datumTools.getDataTools().setRandomSeed(Long.valueOf(SerializationUtil.deserializeAssignmentRight(reader)));
			else if (assignmentLeft.equals("maxThreads"))
				this.maxThreads = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
			else if (!deserializeNext(reader, assignmentLeft))
				return false;
		}
		
		return true;
	}
}
