package ark.model.constraint;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;

public abstract class Constraint<D extends Datum<L>, L> {
	public FeaturizedDataSet<D, L> getSatisfyingSubset(FeaturizedDataSet<D, L> data) {
		return getSatisfyingSubset(data, null);
	}
	
	public FeaturizedDataSet<D, L> getSatisfyingSubset(FeaturizedDataSet<D, L> data, LabelMapping<L> labelMapping) {
		FeaturizedDataSet<D, L> satisfactoryData = new FeaturizedDataSet<D, L>(data.getName(), data.getMaxThreads(), data.getDatumTools(), labelMapping);
		
		for (D datum : data)
			if (isSatisfying(data, datum))
				satisfactoryData.add(datum);
		
		return satisfactoryData;
	}
	
	public abstract boolean isSatisfying(FeaturizedDataSet<D, L> data, D datum);

	// TODO Might want to put this into general deserialization framework of rest of project
	// if want to be able to deserialize features defined in other projects
	// Also, this function is written so that it's only possible to have constraints of the
	// form And(FeatureMatch(...), FeatureMatch(...), ...). Might want to make more generic
	// later
	public static <D extends Datum<L>, L> String fromString(String str, FeaturizedDataSet<D, L> data) {
		
		
		return null;
	}
}
