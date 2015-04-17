package ark.model;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;
import java.util.function.Function;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vector.VectorElement;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.optimization.AdaptiveGradientSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunctionUsingDataSet;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.data.feature.rule.RuleSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.parse.Assignment;
import ark.parse.Assignment.AssignmentTyped;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.OutputWriter;
import ark.util.Pair;

public class SupervisedModelLogistmarGramression<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private boolean weightedLabels = false;
	private double l2 = .00001;
	private double convergenceEpsilon = -1.0;
	private int maxEvaluationConstantIterations = 100000;
	private double maxTrainingExamples = 10000;
	private int batchSize = 100;
	private int evaluationIterations = 500;
	private double classificationThreshold = 0.5;
	private boolean computeTestEvaluations = true;
	private double t = 0.75;
	private RuleSet<D, L> rules;
	private String[] hyperParameterNames = { "t", "rules", "weightedLabels", "l2", "convergenceEpsilon", "maxEvaluationConstantIterations", "maxTrainingExamples", "batchSize", "evaluationIterations", "classificationThreshold", "computeTestEvaluations" };
	
	private Vector u;
	private Map<Integer, String> nonZeroFeatureNamesF_0;
	private Map<Integer, Integer> childToParentFeatureMap; // FIXME This should map to a list of integers to make it so that child can have multiple parents
	private FeaturizedDataSet<D, L> constructedFeatures; // Features constructed through heuristic rules
	private int sizeF_0;

	protected static class NonNegativeAdaptiveGradientSolver extends AdaptiveGradientSolver {
		protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> extends AdaptiveGradientSolver.AbstractBuilder<T> {
			public AbstractBuilder(AbstractStochasticFunction objective, Vector initialPoint) {
				super(objective, initialPoint);
			}

			public NonNegativeAdaptiveGradientSolver build() {
				return new NonNegativeAdaptiveGradientSolver(this);
			}
		}
	
		public static class Builder extends AbstractBuilder<Builder> {
			public Builder(AbstractStochasticFunction objective, Vector initialPoint) {
				super(objective, initialPoint);
			}
	
			 @Override
			 protected Builder self() {
				 return this;
			 }
		}
	
		private NonNegativeAdaptiveGradientSolver(AbstractBuilder<?> builder) {
			super(builder);
		}
		
		@Override
		public void updatePoint() {
			super.updatePoint();
			List<Integer> negativeIndices = new ArrayList<Integer>();
			for (VectorElement element : currentPoint) {
				if (Double.compare(element.value(), 0.0) < 0)
					negativeIndices.add(element.index());
			}
			
			for (Integer negativeIndex : negativeIndices)
				currentPoint.set(negativeIndex, 0.0);
		}
	}
	
	protected class Likelihood extends AbstractStochasticFunctionUsingDataSet<LabeledDataInstance<Vector, Double>> {
		// FIXME this will need to map to a list instead of a range when
		// things are reimplement to account for multiple parents for the same feature
		private Map<Integer, Pair<Integer, Integer>> expandedFeatures; 
		private FeaturizedDataSet<D, L> arkDataSet;
		
		@SuppressWarnings("unchecked")
		public Likelihood(Random random, FeaturizedDataSet<D, L> arkDataSet) {
			this.arkDataSet = arkDataSet;
			this.dataSet = 
					(DataSet<LabeledDataInstance<Vector, Double>>)(DataSet<? extends LabeledDataInstance<Vector, Double>>)
					arkDataSet.makePlataniosDataSet(SupervisedModelLogistmarGramression.this.weightedLabels, 1.0/5.0, true, true);
			
			this.random = random;
			// Features that have had heuristic rules applied mapped to the range of children indices (start-inclusive, end-exclusive)
			this.expandedFeatures = new HashMap<Integer, Pair<Integer, Integer>>();
		}

		@Override
		public Vector estimateGradient(Vector weights, List<LabeledDataInstance<Vector, Double>> dataBatch) {
			double l2 = SupervisedModelLogistmarGramression.this.l2;
			
			Pair<Vector, Vector> uPosNeg = splitPosNeg(weights);
			Vector u_p = uPosNeg.getFirst();
			Vector u_n = uPosNeg.getSecond();
	
			Pair<Vector, Vector> cPosNeg = c(u_p, u_n);
			Vector c_p = cPosNeg.getFirst();
			Vector c_n = cPosNeg.getSecond();
			
			extendDataSet(c_p, c_n);
			
			Map<Integer, Double> mapG_n = new HashMap<Integer, Double>();
			Map<Integer, Double> mapG_p = new HashMap<Integer, Double>();
			
			// NOTE Non-zero elements of gc/gf's' are at f=f' and f\in H^*(f') (if f' is unexpanded, then gc/gf' is 
			// not relevant to element of gradient
			
			Map<Integer, Vector> gC_pp = new HashMap<Integer, Vector>(this.expandedFeatures.size() * 2);
			Map<Integer, Vector> gC_np = new HashMap<Integer, Vector>(this.expandedFeatures.size() * 2);
			Map<Integer, Vector> gC_pn = new HashMap<Integer, Vector>(this.expandedFeatures.size() * 2);
			Map<Integer, Vector> gC_nn = new HashMap<Integer, Vector>(this.expandedFeatures.size() * 2);
			for (Integer expandedFeatureIndex : this.expandedFeatures.keySet()) {
				Pair<Vector, Vector> gCPosNeg_p = gC(expandedFeatureIndex, true, c_p, c_n, u_p, u_n);
				Pair<Vector, Vector> gCPosNeg_n = gC(expandedFeatureIndex, false, c_p, c_n, u_p, u_n);
				gC_pp.put(expandedFeatureIndex, gCPosNeg_p.getFirst());
				gC_np.put(expandedFeatureIndex, gCPosNeg_p.getSecond());
				gC_pn.put(expandedFeatureIndex, gCPosNeg_n.getFirst());
				gC_nn.put(expandedFeatureIndex, gCPosNeg_n.getSecond());
				
				mapG_p.put(expandedFeatureIndex, l2*(c_p.dot(gCPosNeg_p.getFirst())+c_n.dot(gCPosNeg_p.getSecond())));
				mapG_n.put(expandedFeatureIndex, l2*(c_p.dot(gCPosNeg_n.getFirst())+c_n.dot(gCPosNeg_n.getSecond())));
			}
			
			for (VectorElement e_c_p : c_p) {
				if (!this.expandedFeatures.containsKey(e_c_p.index())) {
					mapG_p.put(e_c_p.index(), l2*e_c_p.value());
				}
			}
			
			for (VectorElement e_c_n : c_n) {
				if (!this.expandedFeatures.containsKey(e_c_n.index())) {
					mapG_n.put(e_c_n.index(), l2*e_c_n.value());
				}
			}
			
			for (LabeledDataInstance<Vector, Double> dataInstance : dataBatch) {
				Vector f = dataInstance.features();
				double r = posteriorForDatum(dataInstance, c_p, c_n);
				double y = dataInstance.label();
			
				for (VectorElement e_f : f) {
					if (!this.expandedFeatures.containsKey(e_f.index())) {
						if (!mapG_n.containsKey(e_f.index())) {
							mapG_n.put(e_f.index(), 0.0);
							mapG_p.put(e_f.index(), 0.0);
						}
						
						mapG_n.put(e_f.index(), mapG_n.get(e_f.index()) + e_f.value()*(y-r));
						mapG_p.put(e_f.index(), mapG_p.get(e_f.index()) + e_f.value()*(r-y));
					} 
				}
				
				for (Integer expandedFeatureIndex : this.expandedFeatures.keySet()) {
					Vector gC_pp_fI = gC_pp.get(expandedFeatureIndex);
					Vector gC_np_fI = gC_np.get(expandedFeatureIndex);
					Vector gC_pn_fI = gC_pp.get(expandedFeatureIndex);
					Vector gC_nn_fI = gC_nn.get(expandedFeatureIndex);
				
					if (!mapG_n.containsKey(expandedFeatureIndex)) {
						mapG_n.put(expandedFeatureIndex, 0.0);
						mapG_p.put(expandedFeatureIndex, 0.0);
					}

					mapG_p.put(expandedFeatureIndex, mapG_p.get(expandedFeatureIndex) + f.dot(gC_pp_fI.mult(r-y).addInPlace(gC_np_fI.mult(y-r))));
					mapG_n.put(expandedFeatureIndex, mapG_n.get(expandedFeatureIndex) + f.dot(gC_pn_fI.mult(r-y).addInPlace(gC_nn_fI.mult(y-r))));
				}
			}
			
			return joinPosNeg(new SparseVector(Integer.MAX_VALUE, mapG_p), new SparseVector(Integer.MAX_VALUE, mapG_n));
		}
		
		/*
		 * Compute gradients for elements of c. 
		 * 
		 * FIXME: Note that the current implementation of this function assumes that
		 * the all parents of a feature with index i have indices smaller than i.
		 * This will be true for heuristics where each feature has a single parent,
		 * which is true for most of the heuristics we currently consider.  The only
		 * current exception is the n to n+1 gram heuristic which filters by subspan
		 * (which can go something like "dog -> the dog -> the dog barks" and also
		 * "dog -> dog barks -> the dog barks", but we currently treat "the dog barks"
		 * resulting from two separate heuristic paths as two separate features.
		 * This can be improved in the future by reimplementing the computation of
		 * gC to take account of multiple paths to the same feature.
		 */
		private Pair<Vector, Vector> gC(int wrt_fI, boolean wrt_p, Vector c_p, Vector c_n, Vector u_p, Vector u_n) {
			Map<Integer, Double> gC_p = new HashMap<Integer, Double>();
			Map<Integer, Double> gC_n = new HashMap<Integer, Double>();
			
			// f = f', s = s'
			if (wrt_fI < SupervisedModelLogistmarGramression.this.sizeF_0) {
				// f in F_0
				if (wrt_p) {
					gC_p.put(wrt_fI, 1.0);
				} else {
					gC_n.put(wrt_fI, 1.0);
				}
			} else {
				// f not in F_0
				Integer parentFeatureIndex = SupervisedModelLogistmarGramression.this.childToParentFeatureMap.get(wrt_fI);
				double maxValue = 0.0;
				maxValue = Math.max(maxValue, c_p.get(parentFeatureIndex) - SupervisedModelLogistmarGramression.this.t);
				maxValue = Math.max(maxValue, c_n.get(parentFeatureIndex) - SupervisedModelLogistmarGramression.this.t);
	
				if (Double.compare(maxValue, 0.0) != 0) {
					if (wrt_p) {
						gC_p.put(wrt_fI, maxValue);
					} else {
						gC_n.put(wrt_fI, maxValue);
					}
				}
			}
			
			// f != f', f\in H^*(f')
			Queue<Integer> toVisit = new LinkedList<Integer>();
			Pair<Integer, Integer> childRange = this.expandedFeatures.get(wrt_fI);
			for (int i = childRange.getFirst(); i < childRange.getSecond(); i++)
				toVisit.add(i);
			
			while (!toVisit.isEmpty()) {
				Integer next_fI = toVisit.remove();
				
				Integer parentFeatureIndex = SupervisedModelLogistmarGramression.this.childToParentFeatureMap.get(next_fI);
				double maxValue = 0.0;
				int maxParentFeatureIndex = -1;
				boolean maxP = true;
				
				double value_p = c_p.get(parentFeatureIndex) - SupervisedModelLogistmarGramression.this.t;
				if (value_p > maxValue) {
					maxValue = value_p;
					maxParentFeatureIndex = parentFeatureIndex;
					maxP = true;
				}
				
				double value_n = c_n.get(parentFeatureIndex) - SupervisedModelLogistmarGramression.this.t;
				if (value_n > maxValue) {
					maxValue = value_n;
					maxParentFeatureIndex = parentFeatureIndex;
					maxP = false;
				}
				
				if (maxParentFeatureIndex >= 0) {
					double parentDerivative = (maxP) ? gC_p.get(maxParentFeatureIndex) : gC_n.get(maxParentFeatureIndex);
					gC_p.put(next_fI, u_p.get(next_fI)*parentDerivative);
					gC_n.put(next_fI, u_n.get(next_fI)*parentDerivative);
				}
				
				if (this.expandedFeatures.containsKey(next_fI)) {
					childRange = this.expandedFeatures.get(next_fI);
					for (int i = childRange.getFirst(); i < childRange.getSecond(); i++)
						toVisit.add(i);
				}
			}
			
			return new Pair<Vector, Vector>(new SparseVector(Integer.MAX_VALUE, gC_p), new SparseVector(Integer.MAX_VALUE, gC_n));
		}
		
		// FIXME This should eventually include a mechanism for detecting duplicate features expanded
		// through multiple heuristic paths
		private void extendDataSet(Vector c_p, Vector c_n) {
			Set<Integer> featuresToExpand = getFeaturesToExpand(c_p, c_n);
			RuleSet<D, L> rules = SupervisedModelLogistmarGramression.this.rules;
			
			for (Integer featureToExpand : featuresToExpand) {
				FeaturizedDataSet<D, L> dataSet = (featureToExpand < SupervisedModelLogistmarGramression.this.sizeF_0) ? this.arkDataSet : SupervisedModelLogistmarGramression.this.constructedFeatures;
				// Subtract 1 because bias entry
				int dataSetIndex = (featureToExpand < SupervisedModelLogistmarGramression.this.sizeF_0) ? featureToExpand - 1 : featureToExpand - SupervisedModelLogistmarGramression.this.sizeF_0;
				int featureStartIndex = dataSet.getFeatureStartVocabularyIndex(dataSetIndex);
				Feature<D, L> featureObj = dataSet.getFeatureByVocabularyIndex(dataSetIndex); 
				String featureVocabStr = featureObj.getVocabularyTerm(dataSetIndex - featureStartIndex);
				Map<String, Obj> featureStrAssignment = new TreeMap<String, Obj>();
				featureStrAssignment.put("FEATURE_STR", Obj.stringValue(featureVocabStr));
				
				SupervisedModelLogistmarGramression.this.context.getDatumTools().getDataTools().getOutputWriter()
				.debugWriteln("Expanding feature " + featureToExpand + " (" + featureObj.getReferenceName() + "-" + featureVocabStr + ") c=(" + c_p.get(featureToExpand) + "," + c_n.get(featureToExpand) + ")...");
				
				Map<String, Obj> featureChildObjs = rules.applyRules(featureObj, featureStrAssignment);
				int startVocabularyIndex = SupervisedModelLogistmarGramression.this.sizeF_0 + SupervisedModelLogistmarGramression.this.constructedFeatures.getFeatureVocabularySize();
				int endVocabularyIndex = startVocabularyIndex;
				for (Entry<String, Obj> entry : featureChildObjs.entrySet()) {
					Obj.Function featureChildFunction = (Obj.Function)entry.getValue();
					Feature<D, L> featureChild = this.arkDataSet.getDatumTools().makeFeatureInstance(featureChildFunction.getName(), SupervisedModelLogistmarGramression.this.context);
					featureChild.fromParse(null, featureObj.getReferenceName() + "_" + featureVocabStr + "_" + entry.getKey(), featureChildFunction); // FIXME Throw exception on return false
					featureChild.init(this.arkDataSet); // FIXME Throw exception on false
					
					endVocabularyIndex += featureChild.getVocabularySize();
					
					SupervisedModelLogistmarGramression.this.constructedFeatures.addFeature(featureChild, false); // FIXME Throw exception on false
				}
				
				for (int i = startVocabularyIndex; i < endVocabularyIndex; i++) {
					SupervisedModelLogistmarGramression.this.childToParentFeatureMap.put(i, featureToExpand);
				}
				
				this.expandedFeatures.put(featureToExpand, new Pair<Integer, Integer>(startVocabularyIndex, endVocabularyIndex));
				
				if (endVocabularyIndex != startVocabularyIndex) {
					for (LabeledDataInstance<Vector, Double> datum : this.dataSet) {
						D arkDatum = this.arkDataSet.getDatumById(Integer.valueOf(datum.name()));
						
						Vector extendedDatumValues = SupervisedModelLogistmarGramression.this.
							constructedFeatures.computeFeatureVocabularyRange(arkDatum, 
																			  startVocabularyIndex - SupervisedModelLogistmarGramression.this.sizeF_0, 
																			  endVocabularyIndex - SupervisedModelLogistmarGramression.this.sizeF_0);
						
						datum.features().set(startVocabularyIndex, endVocabularyIndex - 1, extendedDatumValues);
					}
				}
			}
			
		}
		
		private Set<Integer> getFeaturesToExpand(Vector c_p, Vector c_n) {
			Set<Integer> featuresToExpand = new HashSet<Integer>();
			
			for (VectorElement e_p : c_p) {
				if (e_p.index() != 0 && e_p.value() > SupervisedModelLogistmarGramression.this.t && !this.expandedFeatures.containsKey(e_p.index()))
					featuresToExpand.add(e_p.index());
			}
			
			for (VectorElement e_n : c_n) {
				if (e_n.index() != 0 && e_n.value() > SupervisedModelLogistmarGramression.this.t && !this.expandedFeatures.containsKey(e_n.index()))
					featuresToExpand.add(e_n.index());
			}
			
			return featuresToExpand;
		}
	}
	
	public SupervisedModelLogistmarGramression() {
		
	}
	
	public SupervisedModelLogistmarGramression(Context<D, L> context) {
		this.context = context;
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: LogistmarGramression only supports binary classification.");
			return false;
		}
		
		this.u = new SparseVector(Integer.MAX_VALUE);
		this.childToParentFeatureMap = new HashMap<Integer, Integer>();
		this.sizeF_0 = data.getFeatureVocabularySize() + 1; // Add 1 for bias term
		this.constructedFeatures = new FeaturizedDataSet<D, L>("", data.getMaxThreads(), this.context.getDatumTools(), data.getLabelMapping());
		
		List<Double> initEvaluationValues = new ArrayList<Double>();
		for (int i = 0; i < evaluations.size(); i++) {
			initEvaluationValues.add(0.0);
		}
		
		int maximumIterations =  (int)(this.maxTrainingExamples/this.batchSize);	
		
		NumberFormat format = new DecimalFormat("#0.000000");
		
		output.debugWriteln("Logistmar gramression (" + data.getName() + ") training for at most " + maximumIterations + 
				" iterations (maximum " + this.maxTrainingExamples + " examples over size " + this.batchSize + " batches."/*from " + plataniosData.size() + " examples)"*/);
				
		SupervisedModel<D, L> thisModel = this;
		this.u = new NonNegativeAdaptiveGradientSolver.Builder(new Likelihood(data.getDatumTools().getDataTools().makeLocalRandom(), data), this.u)
						.sampleWithReplacement(false)
						.maximumNumberOfIterations(maximumIterations)
						.maximumNumberOfIterationsWithNoPointChange(5)
						.pointChangeTolerance(this.convergenceEpsilon)
						.checkForPointConvergence(true)
						.additionalCustomConvergenceCriterion(new Function<Vector, Boolean>() {
							int iterations = 0;
							int evaluationConstantIterations = 0;
							Map<D, L> prevPredictions = null;
							List<Double> prevEvaluationValues = initEvaluationValues;
							SupervisedModel<D, L> model = thisModel;
							Vector prevU = null;
							
							@Override
							public Boolean apply(Vector weights) {
								this.iterations++;
								
								if (this.iterations % evaluationIterations != 0) {
									this.prevU = u;
									u = weights;
									return false;
								}
								
								double pointChange = weights.sub(this.prevU).norm(VectorNorm.L2_FAST);
								
								String amountDoneStr = format.format(this.iterations/(double)maximumIterations);
								String pointChangeStr = format.format(pointChange);
								String statusStr = data.getName() + " (t=" + t + ", l2=" + l2 + ") #" + iterations + 
										" [" + amountDoneStr + "] -- point-change: " + pointChangeStr + " ";
								
								if (!computeTestEvaluations) {
									output.debugWriteln(statusStr);
									return false;
								}

								this.prevU = u;
								u = weights;
								
								Map<D, L> predictions = classify(testData);
								int labelDifferences = countLabelDifferences(prevPredictions, predictions);
								List<Double> evaluationValues = new ArrayList<Double>();
								for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
									evaluationValues.add(evaluation.evaluate(model, testData, predictions));
								}
								
								statusStr += " predict-diff: " + labelDifferences + "/" + predictions.size() + " ";
								for (int i = 0; i < evaluations.size(); i++) {
									String evaluationName = evaluations.get(i).getGenericName();
									String evaluationDiffStr = format.format(evaluationValues.get(i) - this.prevEvaluationValues.get(i));
									String evaluationValueStr= format.format(evaluationValues.get(i));
									statusStr += evaluationName + " diff: " + evaluationDiffStr + " " + evaluationName + ": " + evaluationValueStr + " ";
								}
								output.debugWriteln(statusStr);
								
								double evaluationDiff = evaluationValues.get(0) - this.prevEvaluationValues.get(0);
								if (Double.compare(evaluationDiff, 0.0) == 0) {
									this.evaluationConstantIterations += evaluationIterations;
								} else {
									this.evaluationConstantIterations = 0;
								}
									
								this.prevPredictions = predictions;
								this.prevEvaluationValues = evaluationValues;
								
								if (maxEvaluationConstantIterations < this.evaluationConstantIterations)
									return true;
								
								return false;
							}
							
							private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
								if (labels1 == null && labels2 != null)
									return labels2.size();
								if (labels1 != null && labels2 == null)
									return labels1.size();
								if (labels1 == null && labels2 == null)
									return 0;
								
								int count = 0;
								for (Entry<D, L> entry: labels1.entrySet()) {
									if (!labels2.containsKey(entry.getKey()) || !entry.getValue().equals(labels2.get(entry.getKey())))
										count++;
								}
								return count;
							}
							
						})
						.batchSize(this.batchSize)
						.stepSize(StochasticSolverStepSize.SCALED)
						.stepSizeParameters(new double[] { 10, 0.75 })
						.useL1Regularization(false)
						.l1RegularizationWeight(0.0)
						.useL2Regularization(false)
						.l2RegularizationWeight(0.0)
						.loggingLevel(0)
						.build()
						.solve(); 

		Pair<Vector, Vector> uPosNeg = splitPosNeg(this.u);
		Vector u_p = uPosNeg.getFirst();
		Vector u_n = uPosNeg.getSecond();
		Set<Integer> nonZeroWeightIndices = new HashSet<Integer>();
		for (VectorElement e_p : u_p) {
			if (e_p.index() == 0 || e_p.index() >= this.sizeF_0)
				continue;
			nonZeroWeightIndices.add(e_p.index() - 1);
		}
		
		for (VectorElement e_n : u_n) {
			if (e_n.index() == 0 || e_n.index() >= this.sizeF_0)
				continue;
			nonZeroWeightIndices.add(e_n.index() - 1);
		}
		
		this.nonZeroFeatureNamesF_0 = data.getFeatureVocabularyNamesForIndices(nonZeroWeightIndices);
		
		output.debugWriteln("Logistmar gramression (" + data.getName() + ") finished training."); 
		
		return true;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();

		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Logistmar Gramression only supports binary classification.");
			return null;
		}
		
		DataSet<PredictedDataInstance<Vector, Double>> plataniosData = data.makePlataniosDataSet(this.weightedLabels, 0.0, false, true);
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>();
		
		Pair<Vector, Vector> uPosNeg = splitPosNeg(this.u);
		Pair<Vector, Vector> cPosNeg = c(uPosNeg.getFirst(), uPosNeg.getSecond());
		
		for (PredictedDataInstance<Vector, Double> plataniosDatum : plataniosData) {
			int datumId = Integer.parseInt(plataniosDatum.name());
			D datum = data.getDatumById(datumId);
			Vector constructedF = this.constructedFeatures.computeFeatureVocabularyRange(datum, 0, this.constructedFeatures.getFeatureVocabularySize());
			plataniosDatum.features().set(this.sizeF_0, this.sizeF_0 + this.constructedFeatures.getFeatureVocabularySize() - 1, constructedF);
			double p = posteriorForDatum(plataniosDatum, cPosNeg.getFirst(), cPosNeg.getSecond());
			Map<L, Double> posterior = new HashMap<L, Double>();
			
			// Offset bias term according to classification threshold so that
			// output posterior p >= 0.5 iff model_p >= classification threshold
			// i.e. log (p/(1-p)) = log (model_p/(1-model_p)) - log (threshold/(1-threshold))
			p = p*(1.0-this.classificationThreshold)/(this.classificationThreshold*(1.0-p)+p*(1.0-this.classificationThreshold));
			
			posterior.put((L)(new Boolean(true)), p);
			posterior.put((L)(new Boolean(false)), 1.0-p);
			
			posteriors.put(datum, posterior);
		}

		return posteriors;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();

		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return null;
		}
		
		DataSet<PredictedDataInstance<Vector, Double>> plataniosData = data.makePlataniosDataSet(this.weightedLabels, 0.0, false, true);
		Map<D, Boolean> predictions = new HashMap<D, Boolean>();
		
		Pair<Vector, Vector> uPosNeg = splitPosNeg(this.u);
		Pair<Vector, Vector> cPosNeg = c(uPosNeg.getFirst(), uPosNeg.getSecond());
		
		for (PredictedDataInstance<Vector, Double> plataniosDatum : plataniosData) {
			int datumId = Integer.parseInt(plataniosDatum.name());
			D datum = data.getDatumById(datumId);
			
			if (this.fixedDatumLabels.containsKey(datum)) {
				predictions.put(datum, (Boolean)this.fixedDatumLabels.get(datum));
				continue;
			}
			
			Vector constructedF = this.constructedFeatures.computeFeatureVocabularyRange(datum, 0, this.constructedFeatures.getFeatureVocabularySize());
			plataniosDatum.features().set(this.sizeF_0, this.sizeF_0 + this.constructedFeatures.getFeatureVocabularySize() - 1, constructedF);
			double p = posteriorForDatum(plataniosDatum, cPosNeg.getFirst(), cPosNeg.getSecond());
			
			if (p >= this.classificationThreshold)
				predictions.put(datum, true);
			else 
				predictions.put(datum, false);
		}
		
		
		return (Map<D, L>)predictions;
	}

	@Override
	public String getGenericName() {
		return "LogistmarGramression";
	}
	
	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("rules"))
			return this.rules.toParse();
		else if (parameter.equals("t"))
			return Obj.stringValue(String.valueOf(this.t));
		else if (parameter.equals("l2"))
			return Obj.stringValue(String.valueOf(this.l2));
		else if (parameter.equals("convergenceEpsilon"))
			return Obj.stringValue(String.valueOf(this.convergenceEpsilon));
		else if (parameter.equals("maxTrainingExamples"))
			return Obj.stringValue(String.valueOf(this.maxTrainingExamples));
		else if (parameter.equals("batchSize"))
			return Obj.stringValue(String.valueOf(this.batchSize));
		else if (parameter.equals("evaluationIterations"))
			return Obj.stringValue(String.valueOf(this.evaluationIterations));
		else if (parameter.equals("maxEvaluationConstantIterations"))
			return Obj.stringValue(String.valueOf(this.maxEvaluationConstantIterations));
		else if (parameter.equals("weightedLabels"))
			return Obj.stringValue(String.valueOf(this.weightedLabels));
		else if (parameter.equals("classificationThreshold"))
			return Obj.stringValue(String.valueOf(this.classificationThreshold));
		else if (parameter.equals("computeTestEvaluations"))
			return Obj.stringValue(String.valueOf(this.computeTestEvaluations));
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("rules"))
			this.rules = this.context.getMatchRuleSet(parameterValue);
		else if (parameter.equals("t"))
			this.t = Double.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("l2"))
			this.l2 = Double.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("convergenceEpsilon"))
			this.convergenceEpsilon = Double.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("maxTrainingExamples"))
			this.maxTrainingExamples = Double.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("batchSize"))
			this.batchSize = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("evaluationIterations"))
			this.evaluationIterations = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("maxEvaluationConstantIterations"))
			this.maxEvaluationConstantIterations = Integer.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("weightedLabels"))
			this.weightedLabels = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("classificationThreshold"))
			this.classificationThreshold = Double.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("computeTestEvaluations"))
			this.computeTestEvaluations = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}
	
	@Override
	public String[] getParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	public SupervisedModel<D, L> makeInstance(Context<D, L> context) {
		return new SupervisedModelLogistmarGramression<D, L>(context);
	}

	@Override
	protected boolean fromParseInternalHelper(AssignmentList internalAssignments) {
		if (!internalAssignments.contains("bias") || !internalAssignments.contains("sizeF_0"))
			return true;
		
		Map<Integer, Double> u_pMap = new HashMap<Integer, Double>(); 
		Map<Integer, Double> u_nMap = new HashMap<Integer, Double>();
		
		this.sizeF_0 = Integer.valueOf(internalAssignments.get("sizeF_0").getValue().toString());
		
		Obj.Array biasArray = (Obj.Array)internalAssignments.get("bias").getValue();
		u_pMap.put(0, Double.valueOf(biasArray.getStr(0)));
		u_nMap.put(0, Double.valueOf(biasArray.getStr(1)));
		
		this.childToParentFeatureMap = new HashMap<Integer, Integer>();
		this.constructedFeatures = new FeaturizedDataSet<D, L>("", 1, this.context.getDatumTools(), null); 
		for (int i = 0; i < internalAssignments.size(); i++) {
			AssignmentTyped assignment = (AssignmentTyped)internalAssignments.get(i);
			if (assignment.getType().equals(Context.FEATURE_STR)) {
				Obj.Function fnObj = (Obj.Function)assignment.getValue();
				Feature<D, L> feature = this.context.getDatumTools().makeFeatureInstance(fnObj.getName(), this.context);
				String referenceName = assignment.getName();
				if (!feature.fromParse(null, referenceName, fnObj))
					return false;
				if (!this.constructedFeatures.addFeature(feature, false))
					return false;
			} else if (assignment.getName().startsWith("u-")) {
				Obj.Array uArr = (Obj.Array)assignment.getValue();
				int uIndex = Integer.valueOf(uArr.getStr(5));
				double u_p = Double.valueOf(uArr.getStr(3));
				double u_n = Double.valueOf(uArr.getStr(4));
				u_pMap.put(uIndex, u_p);
				u_nMap.put(uIndex, u_n);
			} else if (assignment.getName().startsWith("cToP-")) {
				int childIndex = Integer.valueOf(assignment.getName().substring(5));
				int parentIndex = Integer.valueOf(((Obj.Value)assignment.getValue()).getStr());
				this.childToParentFeatureMap.put(childIndex, parentIndex);
			}
		}
		
		Vector u_p = new SparseVector(Integer.MAX_VALUE, u_pMap);
		Vector u_n = new SparseVector(Integer.MAX_VALUE, u_nMap);
		this.u = joinPosNeg(u_p, u_n);
		
		return true;
	}
	
	@Override
	protected AssignmentList toParseInternalHelper(
			AssignmentList internalAssignments) {
		if (this.u == null)
			return internalAssignments;
		
		Pair<Vector, Vector> uPosNeg = splitPosNeg(this.u);
		Vector u_p = uPosNeg.getFirst();
		Vector u_n = uPosNeg.getSecond();
		
		Pair<Vector, Vector> cPosNeg = c(uPosNeg.getFirst(), uPosNeg.getSecond());
		Vector c_p = cPosNeg.getFirst();
		Vector c_n = cPosNeg.getSecond();
		
		Map<Integer, Pair<Double, Double>> cMap = new HashMap<Integer, Pair<Double, Double>>();
		for (VectorElement e_p : u_p) {
			if (e_p.index() == 0) // Skip bias
				continue;
			cMap.put(e_p.index(), new Pair<Double,Double>(c_p.get(e_p.index()), c_n.get(e_p.index())));
		}
		
		for (VectorElement e_n : u_n) {
			if (e_n.index() == 0) // Skip bias
				continue;
			cMap.put(e_n.index(), new Pair<Double,Double>(c_p.get(e_n.index()), c_n.get(e_n.index())));
		}
		
		List<Entry<Integer, Pair<Double, Double>>> cList = new ArrayList<Entry<Integer, Pair<Double, Double>>>(cMap.entrySet());
		Collections.sort(cList, new Comparator<Entry<Integer, Pair<Double, Double>>>() {
			@Override
			public int compare(Entry<Integer, Pair<Double, Double>> c1Entry,
					Entry<Integer, Pair<Double, Double>> c2Entry) {
				double c1 = Math.max(c1Entry.getValue().getFirst(), c1Entry.getValue().getSecond());
				double c2 = Math.max(c2Entry.getValue().getFirst(), c2Entry.getValue().getSecond());
				
				if (c1 > c2)
					return -1;
				else if (c1 < c2)
					return 1;
				else
					return 0;
			} });

		
		internalAssignments.add(Assignment.assignmentTyped(null, Context.VALUE_STR, "sizeF_0", Obj.stringValue(String.valueOf(this.sizeF_0))));
		
		Obj.Array bias = Obj.array(new String[] { String.valueOf(u_p.get(0)), String.valueOf(u_n.get(0)) });
		internalAssignments.add(Assignment.assignmentTyped(null, Context.ARRAY_STR, "bias", bias));
		
		List<String> constructedFeatureVocabulary = this.constructedFeatures.getFeatureVocabularyNames();
		for (Entry<Integer, Pair<Double, Double>> cEntry : cList) {
			boolean constructedFeature = cEntry.getKey() < this.sizeF_0;
			String feature_i = (!constructedFeature) ? this.nonZeroFeatureNamesF_0.get(cEntry.getKey() - 1) : constructedFeatureVocabulary.get(cEntry.getKey() - this.sizeF_0);
			String i = String.valueOf(cEntry.getKey());
			String u_p_i = String.valueOf(u_p.get(cEntry.getKey()));
			String u_n_i = String.valueOf(u_n.get(cEntry.getKey()));
			String c_p_i = String.valueOf(cEntry.getValue().getFirst());
			String c_n_i = String.valueOf(cEntry.getValue().getSecond());
			
			Obj.Array weight = Obj.array(new String[] { feature_i, c_p_i, c_n_i, u_p_i, u_n_i, i });
			internalAssignments.add(Assignment.assignmentTyped(null, Context.ARRAY_STR, "u-" + cEntry.getKey() + ((constructedFeature) ? "-c" : ""), weight));
		}
		
		for (int i = 0; i < this.constructedFeatures.getFeatureCount(); i++) {
			Feature<D, L> feature = this.constructedFeatures.getFeature(i);
			internalAssignments.add(Assignment.assignmentTyped(null, Context.FEATURE_STR, feature.getReferenceName(), feature.toParse()));
		}
	
		for (Entry<Integer, Integer> entry : this.childToParentFeatureMap.entrySet()) {
			internalAssignments.add(Assignment.assignmentTyped(null, Context.VALUE_STR, "cToP-" + entry.getKey(), Obj.stringValue(String.valueOf(entry.getValue()))));
		}
		
		this.nonZeroFeatureNamesF_0 = null; // Assumes convert toParse only once... add back in if memory issues
		
		return internalAssignments;
	}
	
	@Override
	protected <T extends Datum<Boolean>> SupervisedModel<T, Boolean> makeBinaryHelper(
			Context<T, Boolean> context, LabelIndicator<L> labelIndicator,
			SupervisedModel<T, Boolean> binaryModel) {
		return binaryModel;
	}
	
	private Vector joinPosNeg(Vector v_p, Vector v_n) {
		int[] indices = new int[v_p.cardinality() + v_n.cardinality()];
		double[] values = new double[v_p.cardinality() + v_n.cardinality()];
		
		Iterator<VectorElement> i_p = v_p.iterator();
		Iterator<VectorElement> i_n = v_n.iterator();
		
		VectorElement e_p = null;
		VectorElement e_n = null;
		for (int i = 0; i_p.hasNext() || i_n.hasNext() || e_p != null || e_n != null; i++) {
			if (e_p == null && i_p.hasNext())
				e_p = i_p.next();
			if (e_n == null && i_n.hasNext())
				e_n = i_n.next();
			
			// Indicates whether to take next element from e_n or e_p next to keep indices in order
			boolean next_p = (e_n == null) || (e_p != null && e_p.index() <= e_n.index());
			
			if (next_p) {
				indices[i] = e_p.index()*2;
				values[i] = e_p.value();
			} else {
				indices[i] = e_n.index()*2+1;
				values[i] = e_n.value();
			}		
		}

		return new SparseVector(Integer.MAX_VALUE, indices, values);
	}
	
	private Pair<Vector, Vector> splitPosNeg(Vector v) {
		Map<Integer, Double> map_p = new HashMap<Integer, Double>(v.cardinality());
		Map<Integer, Double> map_n = new HashMap<Integer, Double>(v.cardinality());
		
		for (VectorElement e : v) {
			if (e.index() % 2 == 0)
				map_p.put(e.index() / 2, e.value());
			else
				map_n.put((e.index() - 1) / 2 , e.value());
		}
		
		return new Pair<Vector, Vector>(new SparseVector(Integer.MAX_VALUE, map_p), new SparseVector(Integer.MAX_VALUE, map_n));
	}
	
	// FIXME: Note that the current implementation of this function assumes that
	// the all parents of a feature with index i have indices smaller than i.
	// This will be true for heuristics where each feature has a single parent,
	// which is true for most of the heuristics we currently consider.  The only
	// current exception is the n to n+1 gram heuristic which filters by subspan
	// (which can go something like "dog -> the dog -> the dog barks" and also
	// "dog -> dog barks -> the dog barks", but we currently treat "the dog barks"
	// resulting from two separate heuristic paths as two separate features.
	// This can be improved in the future by reimplementing the computation of
	// c to take account of multiple paths to the same feature.
	private Pair<Vector, Vector> c(Vector u_p, Vector u_n) {
		Map<Integer, Double> c_p = new HashMap<Integer, Double>(2*u_p.cardinality());
		Map<Integer, Double> c_n = new HashMap<Integer, Double>(2*u_n.cardinality());
		
		Iterator<VectorElement> i_p = u_p.iterator();
		Iterator<VectorElement> i_n = u_n.iterator();
		
		VectorElement e_p = null;
		VectorElement e_n = null;
		while (i_p.hasNext() || i_n.hasNext() || e_p != null || e_n != null) {
			if (e_p == null && i_p.hasNext())
				e_p = i_p.next();
			if (e_n == null && i_n.hasNext())
				e_n = i_n.next();
			
			boolean next_p = (e_n == null) || (e_p != null && e_p.index() <= e_n.index());
			if (next_p) {
				int index = e_p.index();
				if (index < this.sizeF_0)
					c_p.put(index, e_p.value());
				else {
					double maxValue = 0.0;
					Integer parentIndex = this.childToParentFeatureMap.get(index);
					if (c_p.containsKey(parentIndex))
						maxValue = Math.max(c_p.get(parentIndex) - this.t, maxValue);
					if (c_n.containsKey(parentIndex))
						maxValue = Math.max(c_n.get(parentIndex) - this.t, maxValue);
					
					c_p.put(index, e_p.value()*maxValue);
				}
			} else {
				int index = e_n.index();
				if (index < this.sizeF_0)
					c_n.put(index, e_n.value());
				else {
					double maxValue = 0.0;
					Integer parentIndex = this.childToParentFeatureMap.get(index);
					
					if (c_p.containsKey(parentIndex))
						maxValue = Math.max(c_p.get(parentIndex) - this.t, maxValue);
					if (c_n.containsKey(parentIndex))
						maxValue = Math.max(c_n.get(parentIndex) - this.t, maxValue);
					
					c_n.put(index, e_n.value()*maxValue);
				}
			}
		}
	
		return new Pair<Vector, Vector>(new SparseVector(Integer.MAX_VALUE, c_p), new SparseVector(Integer.MAX_VALUE, c_n));
	}
	
	private double posteriorForDatum(LabeledDataInstance<Vector, Double> dataInstance, Vector c_p, Vector c_n) {
		Vector f = dataInstance.features();	
		double c_pDotF = Math.exp(c_p.dot(f));
		return c_pDotF/(c_pDotF + Math.exp(c_n.dot(f)));
	}
}
