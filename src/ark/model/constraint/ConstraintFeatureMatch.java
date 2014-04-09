package ark.model.constraint;

import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Pattern;

import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;

public class ConstraintFeatureMatch<D extends Datum<L>, L> extends Constraint<D, L> {
		private String featureReference;
		private double minValue;
		private Pattern pattern;
		
		public ConstraintFeatureMatch(String featureReference, double minValue, Pattern pattern) {
			this.featureReference = featureReference;
			this.minValue = minValue;
			this.pattern = pattern;
		}
		
		@Override
		public boolean isSatisfying(FeaturizedDataSet<D, L> data, D datum) {	
			Feature<D, L> feature = data.getFeatureByReferenceName(this.featureReference);
			Map<Integer, Double> featureValues = feature.computeVector(datum); // FIXME Faster to refer to data-set,  but this is fine for now
			Map<Integer, String> vocabulary = feature.getVocabularyForIndices(featureValues.keySet());
			
			for (Entry<Integer, String> entry : vocabulary.entrySet()) {
				if (this.pattern.matcher(entry.getValue()).matches() && 
						featureValues.get(entry.getKey()) >= this.minValue)
					return true;
			}
			
			return false;
		}
}
