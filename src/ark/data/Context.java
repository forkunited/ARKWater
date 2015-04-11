package ark.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.feature.Feature;
import ark.data.feature.fn.Fn;
import ark.data.feature.rule.RuleSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.GridSearch;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.parse.ARKParsable;
import ark.parse.ARKParsableFunction;
import ark.parse.Assignment;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.Pair;

public class Context<D extends Datum<L>, L> extends ARKParsable {
	private enum ObjectType {
		MODEL,
		FEATURE,
		GRID_SEARCH,
		EVALUATION,
		RULE_SET,
		TOKEN_SPAN_FN,
		STR_FN,
		TOKEN_SPAN_STR_FN,
		ARRAY,
		VALUE
	}
	
	public static final String MODEL_STR = "model";
	public static final String FEATURE_STR = "feature";
	public static final String GRID_SEARCH_STR = "gs";
	public static final String EVALUATION_STR = "evaluation";
	public static final String RULE_SET_STR = "rs";
	public static final String TOKEN_SPAN_FN_STR = "ts_fn";
	public static final String STR_FN_STR = "str_fn";
	public static final String TOKEN_SPAN_STR_FN_STR = "ts_str_fn";
	public static final String ARRAY_STR = "array";
	public static final String VALUE_STR = "value";
	
	private Datum.Tools<D, L> datumTools;
	private List<Pair<ObjectType, String>> objNameOrdering;
	private Map<String, SupervisedModel<D, L>> models;
	private Map<String, Feature<D, L>> features;
	private Map<String, GridSearch<D, L>> gridSearches;
	private Map<String, SupervisedModelEvaluation<D, L>> evaluations;
	private Map<String, RuleSet<D, L>> ruleSets;
	private Map<String, Fn<List<TokenSpan>, List<TokenSpan>>> tokenSpanFns;
	private Map<String, Fn<List<String>, List<String>>> strFns;
	private Map<String, Fn<List<TokenSpan>, List<String>>> tokenSpanStrFns;
	private Map<String, List<String>> arrays;
	private Map<String, String> values;
	private int currentReferenceId;
	
	public Context(Datum.Tools<D, L> datumTools) {
		this.datumTools = datumTools;
		this.objNameOrdering = new ArrayList<Pair<ObjectType, String>>();
		this.models = new TreeMap<String, SupervisedModel<D, L>>();
		this.features = new TreeMap<String, Feature<D, L>>();
		this.gridSearches = new TreeMap<String, GridSearch<D, L>>();
		this.evaluations = new TreeMap<String, SupervisedModelEvaluation<D, L>>();
		this.ruleSets = new TreeMap<String, RuleSet<D, L>>();
		this.tokenSpanFns = new TreeMap<String, Fn<List<TokenSpan>, List<TokenSpan>>>();
		this.strFns = new TreeMap<String, Fn<List<String>, List<String>>>();
		this.tokenSpanStrFns = new TreeMap<String, Fn<List<TokenSpan>, List<String>>>();
		this.arrays = new TreeMap<String, List<String>>();
		this.values = new TreeMap<String, String>();
		this.currentReferenceId = 0;
	}
	
	public Context(Datum.Tools<D, L> datumTools, List<Feature<D, L>> features) {
		this(datumTools);
		
		for (int i = 0; i < features.size(); i++) {
			this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.FEATURE, features.get(i).getReferenceName()));
			this.features.put(features.get(i).getReferenceName(), features.get(i));
		}
	}
	
	@Override
	public List<String> getModifiers() {
		return new ArrayList<String>();
	}

	@Override
	public Obj toParse() {
		AssignmentList assignmentList = new AssignmentList();
		for (Pair<ObjectType, String> obj : this.objNameOrdering) {
			if (obj.getFirst() == ObjectType.MODEL) {
				assignmentList.add(Assignment.assignmentTyped(this.models.get(obj.getSecond()).getModifiers(), MODEL_STR, obj.getSecond(), this.models.get(obj.getSecond()).toParse()));
			} else if (obj.getFirst() == ObjectType.FEATURE) {
				assignmentList.add(Assignment.assignmentTyped(this.features.get(obj.getSecond()).getModifiers(), FEATURE_STR, obj.getSecond(), this.features.get(obj.getSecond()).toParse()));
			} else if (obj.getFirst() == ObjectType.GRID_SEARCH) {
				assignmentList.add(Assignment.assignmentTyped(this.gridSearches.get(obj.getSecond()).getModifiers(), GRID_SEARCH_STR, obj.getSecond(), this.gridSearches.get(obj.getSecond()).toParse()));				
			} else if (obj.getFirst() == ObjectType.EVALUATION) {
				assignmentList.add(Assignment.assignmentTyped(this.evaluations.get(obj.getSecond()).getModifiers(), EVALUATION_STR, obj.getSecond(), this.evaluations.get(obj.getSecond()).toParse()));
			} else if (obj.getFirst() == ObjectType.RULE_SET) {
				assignmentList.add(Assignment.assignmentTyped(this.ruleSets.get(obj.getSecond()).getModifiers(), RULE_SET_STR, obj.getSecond(), this.ruleSets.get(obj.getSecond()).toParse()));
			} else if (obj.getFirst() == ObjectType.TOKEN_SPAN_FN) {
				assignmentList.add(Assignment.assignmentTyped(this.tokenSpanFns.get(obj.getSecond()).getModifiers(), TOKEN_SPAN_FN_STR, obj.getSecond(), this.tokenSpanFns.get(obj.getSecond()).toParse()));
			} else if (obj.getFirst() == ObjectType.STR_FN) {
				assignmentList.add(Assignment.assignmentTyped(this.strFns.get(obj.getSecond()).getModifiers(), STR_FN_STR, obj.getSecond(), this.strFns.get(obj.getSecond()).toParse()));	
			} else if (obj.getFirst() == ObjectType.TOKEN_SPAN_STR_FN) {
				assignmentList.add(Assignment.assignmentTyped(this.tokenSpanStrFns.get(obj.getSecond()).getModifiers(), TOKEN_SPAN_STR_FN_STR, obj.getSecond(), this.tokenSpanStrFns.get(obj.getSecond()).toParse()));	
			} else if (obj.getFirst() == ObjectType.ARRAY) {
				assignmentList.add(Assignment.assignmentTyped(new ArrayList<String>(), ARRAY_STR, obj.getSecond(), Obj.array(this.arrays.get(obj.getSecond()))));
			} else if (obj.getFirst() == ObjectType.VALUE) {
				assignmentList.add(Assignment.assignmentTyped(new ArrayList<String>(), VALUE_STR, obj.getSecond(), Obj.stringValue(this.values.get(obj.getSecond()))));
			} else {
				return null;
			}
		}
		
		return assignmentList;
	}

	@Override
	protected boolean fromParseHelper(Obj obj) {
		AssignmentList assignmentList = (AssignmentList)obj;
		
		for (int i = 0; i < assignmentList.size(); i++) {
			Assignment.AssignmentTyped assignment = (Assignment.AssignmentTyped)assignmentList.get(i);
			if (assignment.getType().equals(MODEL_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.MODEL, assignment.getName()));
				if (constructFromParseModel(assignment.getName(), assignment.getValue(), assignment.getModifiers()) == null)
					return false;
			} else if (assignment.getType().equals(FEATURE_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.FEATURE, assignment.getName()));
				if (constructFromParseFeature(assignment.getName(), assignment.getValue(), assignment.getModifiers()) == null)
					return false;
			} else if (assignment.getType().equals(GRID_SEARCH_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.GRID_SEARCH, assignment.getName()));
				if (constructFromParseGridSearch(assignment.getName(), assignment.getValue(), assignment.getModifiers()) == null)
					return false;
			} else if (assignment.getType().equals(EVALUATION_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.EVALUATION, assignment.getName()));
				if (constructFromParseEvaluation(assignment.getName(), assignment.getValue(), assignment.getModifiers()) == null)
					return false;
			} else if (assignment.getType().equals(RULE_SET_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.RULE_SET, assignment.getName()));
				if (constructFromParseRuleSet(assignment.getName(), assignment.getValue(), assignment.getModifiers()) == null)
					return false;
			} else if (assignment.getType().equals(TOKEN_SPAN_FN_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.TOKEN_SPAN_FN, assignment.getName()));
				if (getMatchOrConstructTokenSpanFn(assignment.getName(), assignment.getValue()) == null)
					return false;				
			} else if (assignment.getType().equals(STR_FN_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.STR_FN, assignment.getName()));
				if (getMatchOrConstructStrFn(assignment.getName(), assignment.getValue()) == null)
					return false;
			} else if (assignment.getType().equals(TOKEN_SPAN_STR_FN_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.TOKEN_SPAN_STR_FN, assignment.getName()));
				if (getMatchOrConstructTokenSpanStrFn(assignment.getName(), assignment.getValue()) == null)
					return false;
			} else if (assignment.getType().equals(ARRAY_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.ARRAY, assignment.getName()));
				if (constructFromParseArray(assignment.getName(), assignment.getValue()) == null)
					return false;
			} else if (assignment.getType().equals(VALUE_STR)) {
				this.objNameOrdering.add(new Pair<ObjectType, String>(ObjectType.VALUE, assignment.getName()));
				if (constructFromParseValue(assignment.getName(), assignment.getValue()) == null)
					return false;
			} else {
				this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Invalid object type in context '" + assignment.getType() + "'.");
				return false;
			}
		}

		return true;
	}
	
	/* Generic object matching and construction */
	
	private abstract static interface GenericFunctionRetriever<T extends ARKParsableFunction> {
		List<T> retrieve(String name);
	}
	
	private <T extends ARKParsableFunction> T constructFromParse(List<String> modifiers, String referenceName, Obj obj, Map<String, T> storageMap, GenericFunctionRetriever<T> retriever) {
		if (obj.getObjType() != Obj.Type.FUNCTION) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Invalid object type for function construction (" + obj.getObjType() + ").");
			return null;
		}
		
		Obj.Function fnObj = (Obj.Function)obj;
		
		List<T> possibleGenerics = retriever.retrieve(fnObj.getName());
		for (T possibleGeneric : possibleGenerics) {
			if (possibleGeneric.fromParse(modifiers, referenceName, fnObj)) {
				if (referenceName == null) {				
					storageMap.put(String.valueOf(this.currentReferenceId), possibleGeneric);
					this.currentReferenceId++;
				} else {
					storageMap.put(referenceName, possibleGeneric);
				}
				return possibleGeneric;
			}
		}
	
		this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to construct function (" + fnObj.getName() + ").");
		
		return null;
	}
	
	private <T extends ARKParsableFunction> List<T> getMatches(Obj obj, Map<String, T> storageMap) {
		List<T> matches = new ArrayList<T>();
	
		if (obj.getObjType() == Obj.Type.VALUE) {
			Obj.Value vObj = (Obj.Value)obj;
			if (vObj.getType() == Obj.Value.Type.CURLY_BRACED) {
				if (storageMap.containsKey(vObj.getStr())) {
					matches.add(storageMap.get(vObj.getStr()));
					return matches;
				} else {
					this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to resolve reference variable '" + vObj.getStr() + "'.");
					return null;
				}
			}
		}
		
		for (T item : storageMap.values()) {
			Map<String, Obj> matchMap = item.match(obj);
			if (matchMap.size() > 0)
				matches.add(item);	
		}
		
		return matches;
	}
	
	private <T extends ARKParsableFunction> T getMatch(Obj obj, Map<String, T> storageMap) {
		List<T> matches = getMatches(obj, storageMap);
		
		if (matches.size() > 1) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Expected single match but got " + matches.size() + ".");
			return null;
		} else if (matches.size() == 1) {
			return matches.get(0);
		} else
			return null;
	}
	
	public <T extends ARKParsableFunction> T getMatchOrConstruct(Obj obj, Map<String, T> storageMap, GenericFunctionRetriever<T> retriever) {
		return getMatchOrConstruct(null, null, obj, storageMap, retriever);
	}
	
	private <T extends ARKParsableFunction> T getMatchOrConstruct(List<String> modifiers, String referenceName, Obj obj, Map<String, T> storageMap, GenericFunctionRetriever<T> retriever) {
		List<T> matches = getMatches(obj, storageMap);
		if (matches.size() > 1) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Context expected single match but got " + matches.size() + ".");
			return null;
		} else if (matches.size() == 1) {
			return matches.get(0);
		} else {
			return constructFromParse(modifiers, referenceName, obj, storageMap, retriever);
		}
	}
	
	/* Match and construct token span fns */
	
	private GenericFunctionRetriever<Fn<List<TokenSpan>, List<TokenSpan>>> tokenSpanFnRetriever = new GenericFunctionRetriever<Fn<List<TokenSpan>, List<TokenSpan>>>() {
		@Override
		public List<Fn<List<TokenSpan>, List<TokenSpan>>> retrieve(String name) {
			return Context.this.datumTools.makeTokenSpanFns(name, Context.this);
		}
	};
	
	public Fn<List<TokenSpan>, List<TokenSpan>> getMatchTokenSpanFn(Obj obj) {
		return getMatch(obj, this.tokenSpanFns);
	}
	
	public List<Fn<List<TokenSpan>, List<TokenSpan>>> getMatchesTokenSpanFn(Obj obj) {
		return getMatches(obj, this.tokenSpanFns);
	}
	
	public Fn<List<TokenSpan>, List<TokenSpan>> getMatchOrConstructTokenSpanFn(String referenceName, Obj obj) {
		return getMatchOrConstruct(null, referenceName, obj, this.tokenSpanFns, this.tokenSpanFnRetriever);
	}
	
	public Fn<List<TokenSpan>, List<TokenSpan>> getMatchOrConstructTokenSpanFn(Obj obj) {
		return getMatchOrConstruct(obj, this.tokenSpanFns, this.tokenSpanFnRetriever);
	}
	
	/* Match and construct str fns */
	
	private GenericFunctionRetriever<Fn<List<String>, List<String>>> strFnRetriever = new GenericFunctionRetriever<Fn<List<String>, List<String>>>() {
		@Override
		public List<Fn<List<String>, List<String>>> retrieve(String name) {
			return Context.this.datumTools.makeStrFns(name, Context.this);
		}
	};
	
	public Fn<List<String>, List<String>> getMatchStrFn(Obj obj) {
		return getMatch(obj, this.strFns);
	}
	
	public List<Fn<List<String>, List<String>>> getMatchesStrFn(Obj obj) {
		return getMatches(obj, this.strFns);
	}

	public Fn<List<String>, List<String>> getMatchOrConstructStrFn(String referenceName, Obj obj) {
		return getMatchOrConstruct(null, referenceName, obj, this.strFns, this.strFnRetriever);
	}
	
	public Fn<List<String>, List<String>> getMatchOrConstructStrFn(Obj obj) {
		return getMatchOrConstruct(obj, this.strFns, this.strFnRetriever);
	}
	
	/* Match and construct str fns */
	
	private GenericFunctionRetriever<Fn<List<TokenSpan>, List<String>>> tokenSpanStrFnRetriever = new GenericFunctionRetriever<Fn<List<TokenSpan>, List<String>>>() {
		@Override
		public List<Fn<List<TokenSpan>, List<String>>> retrieve(String name) {
			return Context.this.datumTools.makeTokenSpanStrFns(name, Context.this);
		}
	};
	
	public Fn<List<TokenSpan>, List<String>> getMatchTokenSpanStrFn(Obj obj) {
		return getMatch(obj, this.tokenSpanStrFns);
	}
	
	public List<Fn<List<TokenSpan>, List<String>>> getMatchesTokenSpanStrFn(Obj obj) {
		return getMatches(obj, this.tokenSpanStrFns);
	}

	public Fn<List<TokenSpan>, List<String>> getMatchOrConstructTokenSpanStrFn(String referenceName, Obj obj) {
		return getMatchOrConstruct(null, referenceName, obj, this.tokenSpanStrFns, this.tokenSpanStrFnRetriever);
	}
	
	public Fn<List<TokenSpan>, List<String>> getMatchOrConstructTokenSpanStrFn(Obj obj) {
		return getMatchOrConstruct(obj, this.tokenSpanStrFns, this.tokenSpanStrFnRetriever);
	}
	
	/* Match and construct models */
	
	private GenericFunctionRetriever<SupervisedModel<D, L>> modelRetriever = new GenericFunctionRetriever<SupervisedModel<D, L>>() {
		@Override
		public List<SupervisedModel<D, L>> retrieve(String name) {
			SupervisedModel<D, L> model = Context.this.datumTools.makeModelInstance(name, Context.this);
			List<SupervisedModel<D, L>> models = new ArrayList<SupervisedModel<D, L>>();
			models.add(model);
			return models;
		}
	};

	public SupervisedModel<D, L> getMatchModel(Obj obj) {
		return getMatch(obj, this.models);
	}
	
	public List<SupervisedModel<D, L>> getMatchesModel(Obj obj) {
		return getMatches(obj, this.models);
	}
	
	private SupervisedModel<D, L> constructFromParseModel(String referenceName, Obj obj, List<String> modifiers) {
		return constructFromParse(modifiers, referenceName, obj, this.models, this.modelRetriever);
	}
	
	/* Match and construct evaluations */
	
	private GenericFunctionRetriever<SupervisedModelEvaluation<D, L>> evaluationRetriever = new GenericFunctionRetriever<SupervisedModelEvaluation<D, L>>() {
		@Override
		public List<SupervisedModelEvaluation<D, L>> retrieve(String name) {
			SupervisedModelEvaluation<D, L> evaluation = Context.this.datumTools.makeEvaluationInstance(name, Context.this);
			List<SupervisedModelEvaluation<D, L>> evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
			evaluations.add(evaluation);
			return evaluations;
		}
	};

	public SupervisedModelEvaluation<D, L> getMatchEvaluation(Obj obj) {
		return getMatch(obj, this.evaluations);
	}
	
	public List<SupervisedModelEvaluation<D, L>> getMatchesEvaluation(Obj obj) {
		return getMatches(obj, this.evaluations);
	}
	
	private SupervisedModelEvaluation<D, L> constructFromParseEvaluation(String referenceName, Obj obj, List<String> modifiers) {
		return constructFromParse(modifiers, referenceName, obj, this.evaluations, this.evaluationRetriever);
	}
	
	/* Match and construct features */
	
	private GenericFunctionRetriever<Feature<D, L>> featureRetriever = new GenericFunctionRetriever<Feature<D, L>>() {
		@Override
		public List<Feature<D, L>> retrieve(String name) {
			Feature<D, L> feature = Context.this.datumTools.makeFeatureInstance(name, Context.this);
			List<Feature<D, L>> features = new ArrayList<Feature<D, L>>();
			features.add(feature);
			return features;
		}
	};
	
	private Feature<D, L> constructFromParseFeature(String referenceName, Obj obj, List<String> modifiers) {
		return constructFromParse(modifiers, referenceName, obj, this.features, this.featureRetriever);
	}
	
	public Feature<D, L> getMatchFeature(Obj obj) {
		return getMatch(obj, this.features);
	}
	
	public List<Feature<D, L>> getMatchesFeature(Obj obj) {
		return getMatches(obj, this.features);
	}
	
	public Feature<D, L> getMatchOrConstructFeature(String referenceName, Obj obj) {
		return getMatchOrConstruct(null, referenceName, obj, this.features, this.featureRetriever);
	}
	
	/* Match and construct rule sets */
	
	private GenericFunctionRetriever<RuleSet<D, L>> ruleSetRetriever = new GenericFunctionRetriever<RuleSet<D, L>>() {
		@Override
		public List<RuleSet<D, L>> retrieve(String name) {
			RuleSet<D, L> rules = new RuleSet<D, L>(Context.this);
			List<RuleSet<D, L>> rulesList = new ArrayList<RuleSet<D, L>>();
			rulesList.add(rules);
			return rulesList;
		}
	};
	
	public RuleSet<D, L> getMatchRuleSet(Obj obj) {
		return getMatch(obj, this.ruleSets);
	}
	
	private RuleSet<D, L> constructFromParseRuleSet(String referenceName, Obj obj, List<String> modifiers) {
		return constructFromParse(modifiers, referenceName, obj, this.ruleSets, this.ruleSetRetriever);
	}
	
	/* Match and construct grid searches */
	
	private GenericFunctionRetriever<GridSearch<D, L>> gridSearchRetriever = new GenericFunctionRetriever<GridSearch<D, L>>() {
		@Override
		public List<GridSearch<D, L>> retrieve(String name) {
			GridSearch<D, L> gs = new GridSearch<D, L>(Context.this);
			List<GridSearch<D, L>> gsList = new ArrayList<GridSearch<D, L>>();
			gsList.add(gs);
			return gsList;
		}
	};

	public GridSearch<D, L> getMatchGridSearch(Obj obj) {
		return getMatch(obj, this.gridSearches);
	}
	
	private GridSearch<D, L> constructFromParseGridSearch(String referenceName, Obj obj, List<String> modifiers) {
		return constructFromParse(modifiers, referenceName, obj, this.gridSearches, this.gridSearchRetriever);
	}
	
	/* Match and construct arrays */
	
	public List<String> constructFromParseArray(Obj obj) {
		return constructFromParseArray(null, obj);
	}
	
	public List<String> getMatchArray(Obj obj) {
		if (obj.getObjType() == Obj.Type.VALUE) {
			Obj.Value vObj = (Obj.Value)obj;
			if (vObj.getType() == Obj.Value.Type.CURLY_BRACED) {
				if (this.arrays.containsKey(vObj.getStr())) {
					return this.arrays.get(vObj.getStr());
				} else {
					this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to resolve reference variable '" + vObj.getStr() + "'.");
					return null;
				}
			}
		} else if (obj.getObjType() == Obj.Type.ARRAY) {
			Obj.Array arrayObj = (Obj.Array)obj;
			return arrayObj.toList(this.values);
		} 
		
		return null;
	}
	
	private List<String> constructFromParseArray(String referenceName, Obj obj) {
		if (obj.getObjType() != Obj.Type.ARRAY) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Invalid object type for array construction (" + obj.getObjType() + ").");
			return null;
		}
		
		Obj.Array arrObj = (Obj.Array)obj;
		List<String> array = arrObj.toList(this.values);
		if (array == null) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to construct array '" + ((referenceName != null) ? referenceName : "") + "'");
			return null;
		}
		
		if (referenceName == null) {
			this.arrays.put(String.valueOf(this.currentReferenceId), array);
			this.currentReferenceId++;
		} else {
			this.arrays.put(referenceName, array);
		}

		return array;
	}
	
	/* Match and construct values */
	
	public String getMatchValue(Obj obj) {
		if (obj.getObjType() == Obj.Type.VALUE) {
			Obj.Value vObj = (Obj.Value)obj;
			if (vObj.getType() == Obj.Value.Type.CURLY_BRACED) {
				if (this.values.containsKey(vObj.getStr())) {
					return this.values.get(vObj.getStr());
				} else {
					this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to resolve reference variable '" + vObj.getStr() + "'.");
					return null;
				}
			} else if (vObj.getType() == Obj.Value.Type.STRING) {
				return vObj.getValueStr(this.values);
			}
		}
		
		return null;
	}
	
	public String constructFromParseValue(Obj obj) {
		return constructFromParseValue(null, obj);
	}
	
	private String constructFromParseValue(String referenceName, Obj obj) {
		if (obj.getObjType() != Obj.Type.VALUE) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Invalid object type for value construction (" + obj.getObjType() + ").");
			return null;
		}
		
		Obj.Value vObj = (Obj.Value)obj;
		String value = vObj.getValueStr(this.values);
		if (value == null) {
			this.datumTools.getDataTools().getOutputWriter().debugWriteln("ERROR: Failed to construct value '" + ((referenceName != null) ? referenceName : "") + "'");
			return null;
		}
		
		if (referenceName == null) {
			this.values.put(String.valueOf(this.currentReferenceId), value);
			this.currentReferenceId++;
		} else {
			this.values.put(referenceName, value);
		}

		return value;
	}
	
	/* Add objects */
	
	public synchronized boolean addValue(String name, String value) {
		this.values.put(name, value);
		return true;
	}
	
	/* Get objects */
	
	private <T> List<T> getObjects(Map<String, T> objectMap) {
		return new ArrayList<T>(objectMap.values());
	}
	
	public List<Feature<D, L>> getFeatures() {
		return getObjects(this.features);
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return getObjects(this.evaluations);
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluationsWithModifier(String modifier) {
		List<SupervisedModelEvaluation<D, L>> evaluations = getEvaluations();
		List<SupervisedModelEvaluation<D, L>> withModifier = new ArrayList<SupervisedModelEvaluation<D, L>>(evaluations.size());
		
		for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
			if (evaluation.getModifiers().contains(modifier))
				withModifier.add(evaluation);
		}
		
		return withModifier;
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluationsWithoutModifier(String modifier) {
		List<SupervisedModelEvaluation<D, L>> evaluations = getEvaluations();
		List<SupervisedModelEvaluation<D, L>> withoutModifier = new ArrayList<SupervisedModelEvaluation<D, L>>(evaluations.size());
		
		for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
			if (!evaluation.getModifiers().contains(modifier))
				withoutModifier.add(evaluation);
		}
		
		return withoutModifier;
	}
	
	public List<SupervisedModel<D, L>> getModels() {
		return getObjects(this.models);
	}

	public List<GridSearch<D, L>> getGridSearches() {
		return getObjects(this.gridSearches);
	}
	
	public List<RuleSet<D, L>> getRuleSets() {
		return getObjects(this.ruleSets);
	}
	
	public int getIntValue(String name) {
		return Integer.valueOf(this.values.get(name));
	}

	public double getDoubleValue(String name) {
		return Double.valueOf(this.values.get(name));
	}
	
	public String getStringValue(String name) {
		return this.values.get(name);
	}
	
	/* Other stuff */
	
	public Datum.Tools<D, L> getDatumTools() {
		return this.datumTools;
	}
	 
	public <T extends Datum<Boolean>> Context<T, Boolean> makeBinary(Datum.Tools<T, Boolean> binaryTools, LabelIndicator<L> labelIndicator) {
		Context<T, Boolean> binaryContext = new Context<T, Boolean>(binaryTools);

		for (Pair<ObjectType, String> objName : this.objNameOrdering)
			binaryContext.objNameOrdering.add(objName);
		
		for (Entry<String, List<String>> entry : this.arrays.entrySet())
			binaryContext.arrays.put(entry.getKey(), entry.getValue());
	
		for (Entry<String, String> entry : this.values.entrySet())
			binaryContext.values.put(entry.getKey(), entry.getValue());
		
		for (Entry<String, SupervisedModel<D, L>> entry : this.models.entrySet())
			binaryContext.models.put(entry.getKey(), entry.getValue().makeBinary(binaryContext, labelIndicator));
		
		for (Entry<String, Feature<D, L>> entry : this.features.entrySet())
			binaryContext.features.put(entry.getKey(), entry.getValue().makeBinary(binaryContext, labelIndicator));
		
		for (Entry<String, GridSearch<D, L>> entry : this.gridSearches.entrySet())
			binaryContext.gridSearches.put(entry.getKey(), entry.getValue().makeBinary(binaryContext, labelIndicator));
		
		for (Entry<String, SupervisedModelEvaluation<D, L>> entry : this.evaluations.entrySet())
			binaryContext.evaluations.put(entry.getKey(), entry.getValue().makeBinary(binaryContext, labelIndicator));
		
		for (Entry<String, RuleSet<D, L>> entry : this.ruleSets.entrySet())
			binaryContext.ruleSets.put(entry.getKey(), entry.getValue().makeBinary(binaryContext, labelIndicator));
		
		for (Entry<String, Fn<List<TokenSpan>, List<TokenSpan>>> entry : this.tokenSpanFns.entrySet())
			binaryContext.tokenSpanFns.put(entry.getKey(), entry.getValue());
	
		for (Entry<String, Fn<List<String>, List<String>>> entry : this.strFns.entrySet())
			binaryContext.strFns.put(entry.getKey(), entry.getValue());
		
		for (Entry<String, Fn<List<TokenSpan>, List<String>>> entry : this.tokenSpanStrFns.entrySet())
			binaryContext.tokenSpanStrFns.put(entry.getKey(), entry.getValue());
			
		binaryContext.currentReferenceId = this.currentReferenceId;
		
		return binaryContext;
	}
	
	public Context<D, L> clone() {
		Context<D, L> clone = new Context<D, L>(this.datumTools);
		if (!clone.fromParse(getModifiers(), getReferenceName(), toParse()))
			return null;
		return clone;
	}
	
	public Context<D, L> onlyFeatures() {
		Context<D, L> onlyFeatures = new Context<D, L>(this.datumTools);
		
		for (Pair<ObjectType, String> objName : this.objNameOrdering)
			if (objName.getFirst() == ObjectType.FEATURE)
				onlyFeatures.objNameOrdering.add(objName);
		
		for (Entry<String, Feature<D, L>> entry : this.features.entrySet())
			onlyFeatures.features.put(entry.getKey(), entry.getValue());
		
		return onlyFeatures;
	}
}