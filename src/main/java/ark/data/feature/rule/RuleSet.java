package ark.data.feature.rule;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.parse.ARKParsable;
import ark.parse.ARKParsableFunction;
import ark.parse.Assignment;
import ark.parse.Assignment.AssignmentTyped;
import ark.parse.AssignmentList;
import ark.parse.Obj;

public class RuleSet<D extends Datum<L>, L> extends ARKParsableFunction {
	public static final String RULE_STR = "rule";
	
	private Context<D, L> context;
	private Map<String, Obj.Rule> rules;
	
	public RuleSet(Context<D, L> context) {
		this.context = context;
		this.rules = new HashMap<String, Obj.Rule>();
	}
	
	@Override
	public String[] getParameterNames() {
		return new String[0];
	}

	@Override
	public Obj getParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		return false;
	}

	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {
		Map<String, Obj> contextMap = ((AssignmentList)this.context.toParse()).makeObjMap();
		
		for (int i = 0; i < internalAssignments.size(); i++) {
			AssignmentTyped assignment = (AssignmentTyped)internalAssignments.get(i);
			if (assignment.getType().equals(RULE_STR)) {
				Obj.Rule rule = (Obj.Rule)assignment.getValue().clone();
				rule.getSource().resolveValues(contextMap);
				this.rules.put(assignment.getName(), rule);
			}
		}
		
		return true;
	}

	@Override
	protected AssignmentList toParseInternal() {
		AssignmentList assignmentList = new AssignmentList();
		
		for (Entry<String, Obj.Rule> entry : this.rules.entrySet()) {
			assignmentList.add(
				Assignment.assignmentTyped(null, RULE_STR, entry.getKey(), entry.getValue().clone())
			);
		}

		return assignmentList;
	}

	@Override
	public String getGenericName() {
		return "RuleSet";
	}
	
	public <T extends Datum<Boolean>> RuleSet<T, Boolean> makeBinary(Context<T, Boolean> binaryContext, LabelIndicator<L> labelIndicator) {
		RuleSet<T, Boolean> binaryRuleSet = new RuleSet<T, Boolean>(binaryContext);
		
		binaryRuleSet.referenceName = this.referenceName;
		binaryRuleSet.modifiers = this.modifiers;
		binaryRuleSet.rules = new HashMap<String, Obj.Rule>();
		for (Entry<String, Obj.Rule> entry : this.rules.entrySet()) {
			binaryRuleSet.rules.put(entry.getKey(), (Obj.Rule)entry.getValue().clone());
		}
		
		return binaryRuleSet;
	}
	
	public Map<String, Obj> applyRules(ARKParsable sourceObj) {
		return applyRules(sourceObj.toParse(), null);
	}
	
	public Map<String, Obj> applyRules(ARKParsable sourceObj, Map<String, Obj> extraAssignments) {
		return applyRules(sourceObj.toParse(), extraAssignments);
	}
	
	public Map<String, Obj> applyRules(Obj sourceObj, Map<String, Obj> extraAssignments) {
		Map<String, Obj> objs = new TreeMap<String, Obj>();
	
		for (Entry<String, Obj.Rule> e : this.rules.entrySet()) {
			String ruleName = e.getKey();
			Obj.Rule rule = e.getValue();
			
			Map<String, Obj> matches = sourceObj.match(rule.getSource());
			if (matches.size() > 0) {
				// FIXME this is sort of a hack to allow incrementing numbers in rules... do this more
				// systematically later
				Map<String, Obj> incrementedObjs = new HashMap<String, Obj>();
				for (Entry<String, Obj> entry : matches.entrySet()) {
					if (entry.getValue().getObjType() == Obj.Type.VALUE) {
						Obj.Value vObj = (Obj.Value)entry.getValue();
						if (vObj.getType() == Obj.Value.Type.STRING
								&& vObj.getStr().matches("[0-9]+")) {
							int matchValue = Integer.valueOf(vObj.getStr());
							// FIXME This is a hack to allow checking that the mached number is less than some specified number
							if (entry.getKey().contains("<")) { 
								String[] keyParts = entry.getKey().split("<");
								if (matchValue >= Integer.valueOf(keyParts[1].trim())) {
									matches = new HashMap<String, Obj>();
									break;
								}
							}
							
							incrementedObjs.put(entry.getKey() + "++", Obj.stringValue(String.valueOf(matchValue + 1)));
						}
					}
				}
				
				if (matches.size() == 0) // FIXME This happens when n<k match failed above (it's a hack)
					continue;
				
				matches.putAll(incrementedObjs);
		
				if (extraAssignments != null)
					matches.putAll(extraAssignments);
				
				matches.put("RULE", Obj.stringValue(ruleName));
				
				Obj target = rule.getTarget().clone();
				
				target.resolveValues(matches);
			
				objs.put(ruleName, target);
			}
		}
		
		return objs;
	}
}
