package ark.model.constraint;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.util.SerializationUtil;

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

	// TODO This is half-assed due to lack of time
	// Probably want to put this into general deserialization framework of rest of project
	// if want to be able to deserialize features defined in other projects
	// Also, this function is written so that it's only possible to have constraints of the
	// form And(FeatureMatch(...), FeatureMatch(...), ...). Might want to make more generic
	// later
	public static <D extends Datum<L>, L> Constraint<D, L> fromString(String str) {
		StringReader reader = new StringReader(str);
		List<Constraint<D, L>> constraints = new ArrayList<Constraint<D, L>>();
		
		try {
			SerializationUtil.deserializeGenericName(reader); // Deserialize "And("
			char c = 0;
			while (c != ')') {
				SerializationUtil.deserializeGenericName(reader); // Deserialize "FeatureMatch("
				
				StringBuilder featureReference = new StringBuilder();
				do {
					c = (char)reader.read();
					if (c != ',')
						featureReference = featureReference.append(c);
				} while (c != ',');
				
				StringBuilder minValue = new StringBuilder();
				do {
					c = (char)reader.read();
					if (c != ',')
						minValue = minValue.append(c);
				} while (c != ',');
				
				String pattern = SerializationUtil.deserializeString(reader);
				c = (char)reader.read();
				constraints.add(new ConstraintFeatureMatch<D, L>(featureReference.toString().trim(), 
						Double.parseDouble(minValue.toString().trim()), 
						pattern));
			}
		} catch (IOException e) {
			return null;
		}
		
		Constraint<D, L> currentConstraint = constraints.get(0);
		for (int i = 0; i < constraints.size(); i++) {
			currentConstraint = new ConstraintAnd<D, L>(currentConstraint, constraints.get(i));
		}
		
		return currentConstraint;
	}
}
