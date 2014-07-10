/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of theMess (https://github.com/forkunited/theMess)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy 
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations 
 * under the License.
 */

package ark.model.constraint;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.util.SerializationUtil;

/**
 * Constraint represents an abstract constraint on a featurized data set.  
 * Implementations
 * of children of this class can be used to select subsets of a data set
 * whose features satisfy some criteria. 
 * 
 * The constraints are currently only used by ark.model.SupervisedModelPartition
 * to determine which subsets of a data set should be assigned to which model
 * in the model partition.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class Constraint<D extends Datum<L>, L> {
	public FeaturizedDataSet<D, L> getSatisfyingSubset(FeaturizedDataSet<D, L> data) {
		return getSatisfyingSubset(data, null);
	}
	
	/**
	 * 
	 * @param data
	 * @param labelMapping
	 * @return the subset of the data that satisfies this constraint after the labelMapping
	 * function is applied to labels in the data.
	 */
	public FeaturizedDataSet<D, L> getSatisfyingSubset(FeaturizedDataSet<D, L> data, LabelMapping<L> labelMapping) {
		FeaturizedDataSet<D, L> satisfactoryData = new FeaturizedDataSet<D, L>(data.getName(), data.getMaxThreads(), data.getDatumTools(), labelMapping);
		
		for (D datum : data)
			if (isSatisfying(data, datum))
				satisfactoryData.add(datum);
		
		return satisfactoryData;
	}
	
	/**
	 * 
	 * @param data
	 * @param datum
	 * @return true if datum from data satisfies this constraint
	 */
	public abstract boolean isSatisfying(FeaturizedDataSet<D, L> data, D datum);

	/**
	 * FIXME  This is half-assed due to lack of time.
     * We probably want to put this into general deserialization framework of rest of ARKWater
	 * especially if want to be able to deserialize constraints defined in other projects.
	 * This version is currently very hacked together and depends on constraints being of
	 * the form 'And(FeatureMatch(...), FeatureMatch(...), ...)'.  This form does
	 * not match the form of the strings generated by calling the constraints' toString
	 * methods.
	 * 
	 * @param str
	 * @return the constraint represented by the str.
	 */
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
				c = (char)reader.read();
				constraints.add(new ConstraintFeatureMatch<D, L>(featureReference.toString().trim(), 
						Double.parseDouble(minValue.toString().trim()), 
						pattern));
			}
		} catch (IOException e) {
			return null;
		}
		
		Constraint<D, L> currentConstraint = constraints.get(0);
		if (constraints.size() > 1) {
			for (int i = 1; i < constraints.size(); i++) {
				currentConstraint = new ConstraintAnd<D, L>(currentConstraint, constraints.get(i));
			}
		}
		
		return currentConstraint;
	}
}
