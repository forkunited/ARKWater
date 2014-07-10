package ark.data.feature;
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

import java.util.HashMap;
import java.util.Map;

import ark.data.Gazetteer;
import ark.data.annotation.Datum;

/**
 * FeatureGazetteer computes gazetteer features.  For a datum d, 
 * the feature computes the maximum or
 * minimum of a function f_{S(d)}:G ->R where G is a gazetteer of names,
 * and f_{S(d)} is a function defined for string extractor S used on
 * datum d.  For examples, see
 * the feature types that extend this class under the ark.data.feature
 * package.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public abstract class FeatureGazetteer<D extends Datum<L>, L> extends Feature<D, L> {
	/**
	 * ExtremumType determines whether the minimum or maximum
	 * of the appropriate function should be computed
	 *
	 */
	protected enum ExtremumType {
		Minimum,
		Maximum
	}

	protected FeatureGazetteer.ExtremumType extremumType;
	
	protected Gazetteer gazetteer;
	protected Datum.Tools.StringExtractor<D, L> stringExtractor;
	protected String[] parameterNames = {"gazetteer", "stringExtractor"};
	
	protected abstract double computeExtremum(String str);
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>(1);
		vector.put(0, computeExtremum(datum));
		return vector;
	}

	
	protected double computeExtremum(D datum) {
		String[] strs = this.stringExtractor.extract(datum);
		double extremum = (this.extremumType == FeatureGazetteer.ExtremumType.Maximum) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		for (String str : strs) {
			double curExtremum = computeExtremum(str);
			if ((this.extremumType == FeatureGazetteer.ExtremumType.Maximum && curExtremum > extremum)
					|| (this.extremumType == FeatureGazetteer.ExtremumType.Minimum && curExtremum < extremum))
				extremum = curExtremum;	
		}
		return extremum;
	}
	
	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("gazetteer"))
			return (this.gazetteer == null) ? null : this.gazetteer.getName();
		else if (parameter.equals("stringExtractor"))
			return (this.stringExtractor == null) ? null : this.stringExtractor.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("gazetteer"))
			this.gazetteer = datumTools.getDataTools().getGazetteer(parameterValue);
		else if (parameter.equals("stringExtractor"))
			this.stringExtractor = datumTools.getStringExtractor(parameterValue);
		else 
			return false;
		return true;
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		return true;
	}
	
	@Override
	public String getVocabularyTerm(int index) {
		return null;
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		return true;
	}

	@Override
	public int getVocabularySize() {
		return 1;
	}
}
