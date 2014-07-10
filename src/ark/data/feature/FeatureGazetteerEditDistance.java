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


import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.util.StringUtil;

/**
 * For datum d, string extractor S, and gazetteer G, 
 * FeatureGazetteerEditDistance computes
 * 
 * max_{g\in G} E(g, S(d))
 * 
 * Where E is measures the normalized edit-distance 
 * between g and S(d).
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class FeatureGazetteerEditDistance<D extends Datum<L>, L> extends FeatureGazetteer<D, L> {
	private DataTools.StringPairMeasure editDistanceMeasure;
	
	public FeatureGazetteerEditDistance() {
		this.extremumType = FeatureGazetteer.ExtremumType.Minimum;
		
		this.editDistanceMeasure = new DataTools.StringPairMeasure() {
			public double compute(String str1, String str2) {
				return StringUtil.levenshteinDistance(str1, str2)/((double)(str1.length()+str2.length()));
			}
		};
	}
	
	@Override
	protected double computeExtremum(String str) {
		return this.gazetteer.min(str, this.editDistanceMeasure);
	}

	@Override
	public String getGenericName() {
		return "GazetteerEditDistance";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureGazetteerEditDistance<D, L>();
	}
}
