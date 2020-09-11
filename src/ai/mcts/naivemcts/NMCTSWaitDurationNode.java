/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.mcts.naivemcts;

import ai.mcts.MCTSNode;
import rts.*;
import rts.units.Unit;
import util.Pair;
import util.Sampler;

import java.math.BigInteger;
import java.util.*;

/**
 *
 * @author santi
 */
public class NMCTSWaitDurationNode extends MCTSNode {

    public static final int E_GREEDY = 0;
    public static final int UCB1 = 1;

    static public int DEBUG = 0;

    public static float C = 0.05f;   // exploration constant for UCB1

    boolean forceExplorationOfNonSampledActions = true;
    boolean hasMoreActions = true;
    public CustomPlayerActionGenerator moveGenerator;
    HashMap<BigInteger, NMCTSWaitDurationNode> childrenMap = new LinkedHashMap<>();    // associates action codes with children
    // Decomposition of the player actions in unit actions, and their contributions:
    public List<UnitActionTableEntry> unitActionTable;
    double evaluation_bound;    // this is the maximum positive value that the evaluation function can return
    public BigInteger multipliers[];

    public int waitDuration;

    public NMCTSWaitDurationNode(int maxplayer, int minplayer, GameState a_gs, NMCTSWaitDurationNode a_parent,
                                 double a_evaluation_bound, int a_creation_ID, boolean fensa, int waitDuration) throws Exception {
        parent = a_parent;
        gs = a_gs;
        if (parent==null) depth = 0;
                     else depth = parent.depth+1;     
        evaluation_bound = a_evaluation_bound;
        creation_ID = a_creation_ID;
        forceExplorationOfNonSampledActions = fensa;
        this.waitDuration = waitDuration;
        
        while (gs.winner() == -1 &&
               !gs.gameover() &&
               !gs.canExecuteAnyAction(maxplayer) &&
               !gs.canExecuteAnyAction(minplayer)) {
            gs.cycle();
        }
        if (gs.winner() != -1 || gs.gameover()) {
            type = -1;
        } else if (gs.canExecuteAnyAction(maxplayer)) {
            type = 0;
            moveGenerator = new CustomPlayerActionGenerator(gs, maxplayer, waitDuration);
            actions = new ArrayList<>();
            children = new ArrayList<>();
            unitActionTable = new LinkedList<>();
            multipliers = new BigInteger[moveGenerator.getChoices().size()];
            BigInteger baseMultiplier = BigInteger.ONE;
            int idx = 0;
            for (Pair<Unit, List<UnitAction>> choice : moveGenerator.getChoices()) {
                UnitActionTableEntry ae = new UnitActionTableEntry();
                ae.u = choice.m_a;
                ae.nactions = choice.m_b.size();
                ae.actions = choice.m_b;
                ae.accum_evaluation = new double[ae.nactions];
                ae.visit_count = new int[ae.nactions];
                for (int i = 0; i < ae.nactions; i++) {
                    ae.accum_evaluation[i] = 0;
                    ae.visit_count[i] = 0;
                }
                unitActionTable.add(ae);
                multipliers[idx] = baseMultiplier;
                baseMultiplier = baseMultiplier.multiply(BigInteger.valueOf(ae.nactions));
                idx++;
             }
        } else if (gs.canExecuteAnyAction(minplayer)) {
            type = 1;
            moveGenerator = new CustomPlayerActionGenerator(gs, minplayer, waitDuration);
            actions = new ArrayList<>();
            children = new ArrayList<>();
            unitActionTable = new LinkedList<>();
            multipliers = new BigInteger[moveGenerator.getChoices().size()];
            BigInteger baseMultiplier = BigInteger.ONE;
            int idx = 0;
            for (Pair<Unit, List<UnitAction>> choice : moveGenerator.getChoices()) {
                UnitActionTableEntry ae = new UnitActionTableEntry();
                ae.u = choice.m_a;
                ae.nactions = choice.m_b.size();
                ae.actions = choice.m_b;
                ae.accum_evaluation = new double[ae.nactions];
                ae.visit_count = new int[ae.nactions];
                for (int i = 0; i < ae.nactions; i++) {
                    ae.accum_evaluation[i] = 0;
                    ae.visit_count[i] = 0;
                }
                unitActionTable.add(ae);
                multipliers[idx] = baseMultiplier;
                baseMultiplier = baseMultiplier.multiply(BigInteger.valueOf(ae.nactions));
                idx++;
           }
        } else {
            type = -1;
            System.err.println("NaiveMCTSNode: This should not have happened...");
        }
    }

    
    // Naive Sampling:
    public NMCTSWaitDurationNode selectLeaf(int maxplayer, int minplayer, float epsilon_l, float epsilon_g, float epsilon_0, int global_strategy, int max_depth, int a_creation_ID) throws Exception {
        if (unitActionTable == null) return this;
        if (depth>=max_depth) return this;       
        
        /*
        // DEBUG:
        for(PlayerAction a:actions) {
            for(Pair<Unit,UnitAction> tmp:a.getActions()) {
                if (!gs.getUnits().contains(tmp.m_a)) new Error("DEBUG!!!!");
                boolean found = false;
                for(UnitActionTableEntry e:unitActionTable) {
                    if (e.u == tmp.m_a) found = true;
                }
                if (!found) new Error("DEBUG 2!!!!!");
            }
        } 
        */
        
        if (children.size()>0 && r.nextFloat()>=epsilon_0) {
            // sample from the global MAB:
            NMCTSWaitDurationNode selected = null; // Exploit : Global MAB
            if (global_strategy==E_GREEDY) selected = selectFromAlreadySampledEpsilonGreedy(epsilon_g);
            else if (global_strategy==UCB1) selected = selectFromAlreadySampledUCB1(C);
            return selected.selectLeaf(maxplayer, minplayer, epsilon_l, epsilon_g, epsilon_0, global_strategy, max_depth, a_creation_ID);
        }  else {
            // Explore : using Local MABs
            // sample from the local MABs (this might recursively call "selectLeaf" internally):
            return selectLeafUsingLocalMABs(maxplayer, minplayer, epsilon_l, epsilon_g, epsilon_0, global_strategy, max_depth, a_creation_ID);
        }
    }
   

    
    public NMCTSWaitDurationNode selectFromAlreadySampledEpsilonGreedy(float epsilon_g) throws Exception {
        if (r.nextFloat()>=epsilon_g) { // Exploit
            NMCTSWaitDurationNode best = null;
            for(MCTSNode child:children) {
                if (type==0) {
                    // max node:
                    if (best==null || (child.accum_evaluation/child.visit_count)>(best.accum_evaluation/best.visit_count)) {
                        best = (NMCTSWaitDurationNode)child;
                    }                    
                } else {
                    // min node:
                    if (best==null || (child.accum_evaluation/child.visit_count)<(best.accum_evaluation/best.visit_count)) {
                        best = (NMCTSWaitDurationNode)child;
                    }                                        
                }
            }

            return best;
        } else { // Explore
            // choose one at random from the ones seen so far:
            NMCTSWaitDurationNode best = (NMCTSWaitDurationNode)children.get(r.nextInt(children.size()));
            return best;
        }
    }
    
    
    public NMCTSWaitDurationNode selectFromAlreadySampledUCB1(float C) throws Exception {
        NMCTSWaitDurationNode best = null;
        double bestScore = 0;
        for(MCTSNode child:children) {
            double exploitation = ((double)child.accum_evaluation) / child.visit_count;
            double exploration = Math.sqrt(Math.log((double)visit_count)/child.visit_count);
            if (type==0) {
                // max node:
                exploitation = (evaluation_bound + exploitation)/(2*evaluation_bound);
            } else {
                exploitation = (evaluation_bound - exploitation)/(2*evaluation_bound);
            }
    //            System.out.println(exploitation + " + " + exploration);

            double tmp = C*exploitation + exploration;            
            if (best==null || tmp>bestScore) {
                best = (NMCTSWaitDurationNode)child;
                bestScore = tmp;
            }
        }
        
        return best;
    }    
    
    
    public NMCTSWaitDurationNode selectLeafUsingLocalMABs(int maxplayer, int minplayer, float epsilon_l, float epsilon_g, float epsilon_0, int global_strategy, int max_depth, int a_creation_ID) throws Exception {
        PlayerAction playerAction;
        BigInteger actionCode;

        // For each unit, rank the unitActions according to preference:
        List<double []> distributions = new LinkedList<>();
        List<Integer> notSampledYet = new LinkedList<>();

        for (UnitActionTableEntry entry : unitActionTable) {

            double [] dist = new double[entry.nactions];
            int bestIdx = -1;
            double bestEvaluation = 0;
            int visits = 0;

            for (int i = 0; i < entry.nactions; i++) {

                if (type == 0) {
                    // max node:
                    // Find the index value of the action with the highest reward, in case all actions were visited.
                    // In case of an unvisited action, its index is returned as the best index, to favour exploration.
                    if (bestIdx == -1 || // the initial case
                       (visits != 0 && entry.visit_count[i] == 0) || // unvisited action case
                       (visits != 0 && (entry.accum_evaluation[i] / entry.visit_count[i]) > bestEvaluation)) { // better action case

                       bestIdx = i;

                       if (entry.visit_count[i] > 0)
                           // better action case
                           bestEvaluation = (entry.accum_evaluation[i] / entry.visit_count[i]);
                       else
                           // unvisited action case
                           bestEvaluation = 0;

                       // The last action visit count.
                       visits = entry.visit_count[i];
                    }
                } else {
                    // min node:
                    if (bestIdx==-1 || 
                        (visits!=0 && entry.visit_count[i]==0) ||
                        (visits!=0 && (entry.accum_evaluation[i]/entry.visit_count[i])<bestEvaluation)) {
                        bestIdx = i;
                        if (entry.visit_count[i]>0) bestEvaluation = (entry.accum_evaluation[i]/entry.visit_count[i]);
                                             else bestEvaluation = 0;
                        visits = entry.visit_count[i];
                    }
                }

                dist[i] = epsilon_l/entry.nactions;
            }

            if (entry.visit_count[bestIdx] != 0) { // Amplify dist value of the bestIdx unitAction, if it was visited at least once.
                dist[bestIdx] = (1 - epsilon_l) + (epsilon_l / entry.nactions);
            } else {
                // Turn all dist values of actions visited at least once to zero. Because the bestIdx was non-visited
                // Therefore, any visited action should be visited less, and focus should shift to unvisited actions
                // This forces the exploration of non-visited action
                if (forceExplorationOfNonSampledActions) {
                    for (int j = 0; j < dist.length; j++)
                        if (entry.visit_count[j] > 0) dist[j] = 0;
                }
            }

            if (DEBUG>=3) {
                System.out.print("[ ");
                for(int i = 0;i<entry.nactions;i++) System.out.print("(" + entry.visit_count[i] + "," + entry.accum_evaluation[i]/entry.visit_count[i] + ")");
                System.out.println("]");
                System.out.print("[ ");
                for (double v : dist) System.out.print(v + " ");
                System.out.println("]");
            }

            notSampledYet.add(distributions.size()); // expands the list with the index of the current dist.
            distributions.add(dist);
        }

        // Select the best combination that results in a valid playerAction by epsilon-greedy sampling:

        // Compute the resource usage of the unit-actions in the current game state.
        ResourceUsage base_ru = new ResourceUsage();
        for (Unit u : gs.getUnits()) {
            UnitAction ua = gs.getUnitAction(u);
            if (ua != null) {
                ResourceUsage ru = ua.resourceUsage(u, gs.getPhysicalGameState());
                base_ru.merge(ru);
            }
        }

        playerAction = new PlayerAction();
        playerAction.setResourceUsage(base_ru.clone());
        actionCode = BigInteger.ZERO;

        while (!notSampledYet.isEmpty()) {

            // remove an index at random.
            int i = notSampledYet.remove(r.nextInt(notSampledYet.size()));

            try {
                // Search for a unit action.
                UnitActionTableEntry ate = unitActionTable.get(i);
                int code;
                UnitAction ua;
                ResourceUsage r2;

                // Debug
//                System.out.println(ate.actions.get(ate.nactions - 1));

                // try one at random:
                double [] distribution = distributions.get(i);
                code = Sampler.weighted(distribution);
                ua = ate.actions.get(code);
                r2 = ua.resourceUsage(ate.u, gs.getPhysicalGameState());

                // in case the unit action is not consistent with the playerAction.
                if (!playerAction.getResourceUsage().consistentWith(r2, gs)) {
                    // sample at random, eliminating the ones that have not worked so far:
                    List<Double> dist_l = new ArrayList<>(); // distribution converted to list
                    List<Integer> dist_outputs = new ArrayList<>(); // indices of actions and distribution items

                    for (int j = 0 ; j < distribution.length ; j++) { // Conversion
                        dist_l.add(distribution[j]);
                        dist_outputs.add(j);
                    }

                    do {
                        int idx = dist_outputs.indexOf(code);  // determine the index of code in the list
                        dist_l.remove(idx);  // remove the inconsistent action code from dist list, indexed at idx.
                        dist_outputs.remove(idx);

                        code = (Integer)Sampler.weighted(dist_l, dist_outputs); // get a new unit action code
                        ua = ate.actions.get(code); // get the new unit action and its resource usage.
                        r2 = ua.resourceUsage(ate.u, gs.getPhysicalGameState());                            
                    } while (!playerAction.getResourceUsage().consistentWith(r2, gs));
                }

                // DEBUG code:
                if (gs.getUnit(ate.u.getID())==null) throw new Error("Issuing an action to an inexisting unit!!!");
               

                playerAction.getResourceUsage().merge(r2);
                playerAction.addUnitAction(ate.u, ua);

                actionCode = actionCode.add(BigInteger.valueOf(code).multiply(multipliers[i]));

            } catch(Exception e) {
                e.printStackTrace();
            }
        }   

        NMCTSWaitDurationNode node = childrenMap.get(actionCode);
        if (node == null) {
            actions.add(playerAction);
            GameState gs2 = gs.cloneIssue(playerAction);
            NMCTSWaitDurationNode newNode = new NMCTSWaitDurationNode(maxplayer, minplayer, gs2.clone(), this, evaluation_bound,
                    a_creation_ID, forceExplorationOfNonSampledActions, waitDuration);
            childrenMap.put(actionCode,newNode);
            children.add(newNode);
            return newNode;
        }

        return node.selectLeaf(maxplayer, minplayer, epsilon_l, epsilon_g, epsilon_0, global_strategy, max_depth, a_creation_ID);
    }
    
    
    public UnitActionTableEntry getActionTableEntry(Unit u) {
        for (UnitActionTableEntry e : unitActionTable) {
            if (e.u == u) return e;
        }
        throw new Error("Could not find Action Table Entry!");
    }


    public void propagateEvaluation(double evaluation, NMCTSWaitDurationNode child) {
        accum_evaluation += evaluation;
        visit_count++;
        
//        if (child!=null) System.out.println(evaluation);

        // update the unitAction table:
        if (child != null) {
            int idx = children.indexOf(child);
            PlayerAction pa = actions.get(idx);

            for (Pair<Unit, UnitAction> ua : pa.getActions()) {
                UnitActionTableEntry actionTable = getActionTableEntry(ua.m_a);
                idx = actionTable.actions.indexOf(ua.m_b);

                if (idx==-1) {
                    System.out.println("Looking for action: " + ua.m_b);
                    System.out.println("Available actions are: " + actionTable.actions);
                }
                
                actionTable.accum_evaluation[idx] += evaluation;
                actionTable.visit_count[idx]++;
            }
        }

        if (parent != null) {
            ((NMCTSWaitDurationNode)parent).propagateEvaluation(evaluation, this);
        }
    }

    public void printUnitActionTable() {
        for (UnitActionTableEntry uat : unitActionTable) {
            System.out.println("Actions for unit " + uat.u);
            for (int i = 0; i < uat.nactions; i++) {
                System.out.println("   " + uat.actions.get(i) + " visited " + uat.visit_count[i] + " with average evaluation " + (uat.accum_evaluation[i] / uat.visit_count[i]));
            }
        }
    }    
}
