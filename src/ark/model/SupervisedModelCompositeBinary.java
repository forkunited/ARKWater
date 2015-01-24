package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.ThreadMapper;
import ark.util.ThreadMapper.Fn;

public class SupervisedModelCompositeBinary<T extends Datum<Boolean>, D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private List<SupervisedModel<T, Boolean>> binaryModels;
	private List<LabelIndicator<L>> labelIndicators;
	private Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator;
	private Datum.Tools<T, Boolean> binaryTools;
	
	public SupervisedModelCompositeBinary(List<SupervisedModel<T, Boolean>> binaryModels, List<LabelIndicator<L>> labelIndicators, Datum.Tools<T, Boolean> binaryTools, Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		this.binaryModels = binaryModels;
		this.labelIndicators = labelIndicators;
		this.binaryTools = binaryTools;
		this.inverseLabelIndicator = inverseLabelIndicator;
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data,
			FeaturizedDataSet<D, L> testData,
			List<SupervisedModelEvaluation<D, L>> evaluations) {
		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(final FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> p = new HashMap<D, Map<L, Double>>();
		final FeaturizedDataSet<T, Boolean> binaryData = (FeaturizedDataSet<T, Boolean>)data.makeBinaryDataSet(this.binaryTools);
		
		ThreadMapper<SupervisedModel<T, Boolean>, Map<T, Map<Boolean, Double>>> threads 
		= new ThreadMapper<SupervisedModel<T, Boolean>, Map<T, Map<Boolean, Double>>>(
				new Fn<SupervisedModel<T, Boolean>, Map<T, Map<Boolean, Double>>>() {
					@Override
					public Map<T, Map<Boolean, Double>> apply(
							SupervisedModel<T, Boolean> model) {
						return model.posterior(binaryData);
					}
				}
			);
		
		List<Map<T, Map<Boolean, Double>>> binaryP = threads.run(this.binaryModels, data.getMaxThreads());
		
		for (Entry<T, Map<Boolean, Double>> entry : binaryP.get(0).entrySet()) {
			Map<String, Double> indicatorWeights = new HashMap<String, Double>();
			for (int i = 0; i < binaryP.size(); i++) {
				indicatorWeights.put(this.labelIndicators.toString(), entry.getValue().get(true));
			}
			
			L label = this.inverseLabelIndicator.label(indicatorWeights);
			D datum = data.getDatumById(entry.getKey().getId());
			
			Map<L, Double> datumP = new HashMap<L, Double>();
			datumP.put(label, 1.0);
			p.put(datum, datumP);
		}
		
		return p;
	}
	
	@Override
	public String[] getParameterNames() {
		return new String[0];
	}

	@Override
	public String getParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return null;
	}

	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		return true;
	}

	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		return true;
	}

	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		return true;
	}

	@Override
	public String getGenericName() {
		return "CompositeBinary";
	}
}
