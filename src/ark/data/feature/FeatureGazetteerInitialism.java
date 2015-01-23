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

package ark.data.feature;

import java.util.Arrays;
import java.util.List;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.Pair;
import ark.util.StringUtil;

/**
 * For datum d, string extractor S, and gazetteer G, 
 * FeatureGazetteerInitialism computes
 * 
 * max_{g\in G} 1(S(d) is an initialism for g)
 * 
 * An 'allowPrefix' parameter determines whether S(d) must
 * be a full initialism, or just a partial (prefix) initialism.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 *
 */
public class FeatureGazetteerInitialism<D extends Datum<L>, L> extends FeatureGazetteer<D, L> {
	private DataTools.StringPairMeasure initialismMeasure;
	private boolean allowPrefix;
	
	public FeatureGazetteerInitialism() {
		this.extremumType = FeatureGazetteer.ExtremumType.Maximum;
		
		this.initialismMeasure = new DataTools.StringPairMeasure() {
			public double compute(String str1, String str2) {
				if (StringUtil.isInitialism(str1, str2, allowPrefix))
					return 1.0;
				else
					return 0.0;
			}
		};
		
		this.allowPrefix = false;
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 1);
		this.parameterNames[this.parameterNames.length - 1] = "allowPrefix";
	}
	
	@Override
	protected Pair<List<Pair<String,Double>>, Double> computeExtremum(String str) {
		return this.gazetteer.max(str, this.initialismMeasure);
	}

	@Override
	public String getGenericName() {
		return "GazetteerInitialism";
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureGazetteerInitialism<D, L>();
	}
	
	@Override
	public String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("allowPrefix"))
			return String.valueOf(this.allowPrefix);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("allowPrefix"))
			this.allowPrefix = Boolean.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}
}
