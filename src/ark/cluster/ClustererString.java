package ark.cluster;

import java.util.List;

import ark.data.DataTools;

public abstract class ClustererString extends Clusterer<String> {
	protected DataTools.StringTransform cleanFn;
	
	public ClustererString(DataTools.StringTransform cleanFn) {
		this.cleanFn = cleanFn;
	}
	
	@Override
	public List<String> getClusters(String obj) {
		if (this.cleanFn != null)
			obj = this.cleanFn.transform(obj);
		return getClustersHelper(obj);
	}
	
	protected abstract List<String> getClustersHelper(String obj);
}
