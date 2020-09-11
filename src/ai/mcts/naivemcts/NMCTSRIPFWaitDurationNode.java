package ai.mcts.naivemcts;

import aes.nmcts.UnitActionsTableElement;
import rts.*;
import rts.units.Unit;
import util.Pair;
import util.Sampler;

import java.math.BigInteger;
import java.util.*;

public class NMCTSRIPFWaitDurationNode {

    // Global strategies
    public static final int EPSILON_GREEDY = 0;
    public static final int UCB1 = 1;

    public static float C = 0.05f; // UCB1 Exploration Constant.
    public static Random random = new Random();

    // Typical MCTS Node properties ************************************************
    private int type; // 0 : max, 1 : min, -1 : terminal
    private NMCTSRIPFWaitDurationNode parent; // the node's parent, if any
    private GameState gameState; // the associated game state
    private int depth = 0; // the node's depth in the tree
    private List<PlayerAction> actions; // the list of outbound actions
    private List<NMCTSRIPFWaitDurationNode> children; // the list of this node's children
    private double accumulatedEvaluation = 0; // the accumulated evaluation of this node
    private int visitCount = 0; // the visit count
    private CustomPlayerActionGenerator actionGenerator; // the action generator object

    // NaïveMCTS Specific properties ***********************************************
    private boolean exploreNonSampledActions = true; // to force the exploration of unvisited unit actions
    private HashMap<BigInteger, NMCTSRIPFWaitDurationNode> childrenMap = new LinkedHashMap<>(); // Maps a binInteger to a node
    private List<UnitActionsTableElement> unitActionsTable; // Each unit's actions and their evaluation and visit count
    private BigInteger [] multipliers; // For action code calculation

    private int nodeID = -1;
    private int waitDuration;

    private int inactionsCounter = 0;
    List<BigInteger> rejectedActions = new LinkedList<>();

    // The Constructor. ************************************************************

    /**
     *
     * @param player
     * @param gameState
     * @param parent
     * @param nodeID
     * @param exploreNonSampledActions
     * @throws Exception
     */
    public NMCTSRIPFWaitDurationNode(int player, GameState gameState, NMCTSRIPFWaitDurationNode parent, int nodeID,
                                     boolean exploreNonSampledActions, int waitDuration) throws Exception {
        this.parent = parent;
        this.gameState = gameState;
        this.nodeID = nodeID;
        this.exploreNonSampledActions = exploreNonSampledActions;

        if (this.parent == null) depth = 0; // depth calculation
        else depth = this.parent.depth + 1;
        this.waitDuration = waitDuration;

        // Taking account for durative actions :
        while (this.gameState.winner() == -1 &&
               !this.gameState.gameover() &&
               !this.gameState.canExecuteAnyAction(player) &&
               !this.gameState.canExecuteAnyAction(1 - player))
            this.gameState.cycle();

        // Type, and other parameters initialization
        if (this.gameState.winner() != -1 || this.gameState.gameover()) // Terminal node
            type = -1;
        else if (this.gameState.canExecuteAnyAction(player)) { // Searching player node
            type = 0;
            actionGenerator = new CustomPlayerActionGenerator(this.gameState, player, waitDuration);
            actions = new ArrayList<>();
            children = new ArrayList<>();
            unitActionsTable = new LinkedList<>();
            multipliers = new BigInteger[actionGenerator.getChoices().size()];

            BigInteger baseMultiplier = BigInteger.ONE;
            int index = 0;
            for (Pair<Unit, List<UnitAction>> actionChoices : actionGenerator.getChoices()) {
                UnitActionsTableElement unitActionsElement = new UnitActionsTableElement();
                unitActionsElement.unit = actionChoices.m_a;
                unitActionsElement.actions = actionChoices.m_b;
                unitActionsElement.actionCount = actionChoices.m_b.size();
                unitActionsElement.accumulatedEvaluation = new double[unitActionsElement.actionCount];
                unitActionsElement.visitsCount = new int[unitActionsElement.actionCount];

                for (int i = 0; i < unitActionsElement.actionCount; i++) {
                    unitActionsElement.accumulatedEvaluation[i] = 0;
                    unitActionsElement.visitsCount[i] = 0;
                }

                unitActionsTable.add(unitActionsElement);
                multipliers[index] = baseMultiplier;
                baseMultiplier = baseMultiplier.multiply(BigInteger.valueOf(unitActionsElement.actionCount));
                index++;
            }

        } else if (this.gameState.canExecuteAnyAction(1 - player)) { // Opponent node
            type = 1;
            actionGenerator = new CustomPlayerActionGenerator(this.gameState, 1 - player, waitDuration);
            actions = new ArrayList<>();
            children = new ArrayList<>();
            unitActionsTable = new LinkedList<>();
            multipliers = new BigInteger[actionGenerator.getChoices().size()];

            BigInteger baseMultiplier = BigInteger.ONE;
            int index = 0;
            for (Pair<Unit, List<UnitAction>> actionChoices : actionGenerator.getChoices()) {
                UnitActionsTableElement unitActionsElement = new UnitActionsTableElement();
                unitActionsElement.unit = actionChoices.m_a;
                unitActionsElement.actions = actionChoices.m_b;
                unitActionsElement.actionCount = actionChoices.m_b.size();
                unitActionsElement.accumulatedEvaluation = new double[unitActionsElement.actionCount];
                unitActionsElement.visitsCount = new int[unitActionsElement.actionCount];

                for (int i = 0; i < unitActionsElement.actionCount; i++) {
                    unitActionsElement.accumulatedEvaluation[i] = 0;
                    unitActionsElement.visitsCount[i] = 0;
                }

                unitActionsTable.add(unitActionsElement);
                multipliers[index] = baseMultiplier;
                baseMultiplier = baseMultiplier.multiply(BigInteger.valueOf(unitActionsElement.actionCount));
                index++;
            }
        } else {
            type = -1;
            System.err.println("This should not have happened ...");
        }
    }

    /**
     *
     * @param player
     * @param epsilon0
     * @param epsilonGlobal
     * @param epsilonLocal
     * @param globalStrategy
     * @param maxDepth
     * @param nodeID
     * @param evaluationBound
     * @return
     * @throws Exception
     */
    public NMCTSRIPFWaitDurationNode selectLeaf(int player, float epsilon0, float epsilonGlobal, float epsilonLocal, int globalStrategy,
                                                int maxDepth, int nodeID, double evaluationBound, int maxAllowedInactions) throws Exception {
        // Return the current node, if unitActionsTable was not initialized (terminal node)
        // or in case the maximum depth has been reached.
        if (unitActionsTable == null || depth >= maxDepth)
            return this;

        // If the node has children, we can proceed with exploitation.
        if (children.size() > 0 && random.nextFloat() >= epsilon0) {
            // Sample from the global MAB. Exploit.
            NMCTSRIPFWaitDurationNode selected = null;
            if (globalStrategy == EPSILON_GREEDY) selected = selectFromGlobalMABEpsilonGreedy(epsilonGlobal);
            else if (globalStrategy == UCB1) selected = selectFromGlobalMABUCB1(C, evaluationBound);
            return selected.selectLeaf(player, epsilon0, epsilonGlobal, epsilonLocal, globalStrategy, maxDepth, nodeID,
                    evaluationBound, maxAllowedInactions);
        }
        // If there are no children, we must first add children through the local MAB
        else
            // Sample from the local MABs. Explore. (Construct a new player action)
            return selectFromLocalMAB(player, epsilon0, epsilonGlobal, epsilonLocal, globalStrategy, maxDepth, nodeID,
                                      evaluationBound, maxAllowedInactions);
    }

    /**
     *
     * @param epsilonGlobal
     * @return
     */
    private NMCTSRIPFWaitDurationNode selectFromGlobalMABEpsilonGreedy(float epsilonGlobal) {

        NMCTSRIPFWaitDurationNode best = null;

        if (random.nextFloat() >= epsilonGlobal) { // Exploit : choose the best child.
            for (NMCTSRIPFWaitDurationNode child : children) {
                if (type == 0) { // Max Node
                    if (best == null ||
                       (child.accumulatedEvaluation / child.visitCount) > (best.accumulatedEvaluation / best.visitCount))
                        best = child;
                } else {  // Min node
                    if (best == null ||
                       (child.accumulatedEvaluation / child.visitCount) < (best.accumulatedEvaluation / best.visitCount))
                        best = child;
                }
            }
        } else // Explore : Choose a child at random.
            best = children.get(random.nextInt(children.size()));

        return best;
    }

    /**
     *
     * @param C
     * @param evaluationBound
     * @return
     */
    private NMCTSRIPFWaitDurationNode selectFromGlobalMABUCB1(float C, double evaluationBound) {
        NMCTSRIPFWaitDurationNode best = null;
        double bestScore = 0;
        for (NMCTSRIPFWaitDurationNode child : children) {
            // Compute the exploitation and exploration terms for each child.
            double exploitationTerm = child.accumulatedEvaluation / child.visitCount;
            double explorationTerm = Math.sqrt(Math.log((double) visitCount / child.visitCount));
            if (type == 0) // Max node :
                exploitationTerm = (evaluationBound + exploitationTerm) / (2 * evaluationBound);
            else
                exploitationTerm = (evaluationBound - exploitationTerm) / (2 * evaluationBound);

            double evaluation = exploitationTerm + C * explorationTerm;
            if (best == null || evaluation > bestScore) {
                best = child;
                bestScore = evaluation;
            }
        }

        // Returns the child with the highest UCB1 value.
        return best;
    }

    /**
     * Constructs a player action using the local MABs
     * @return
     */
    private NMCTSRIPFWaitDurationNode selectFromLocalMAB(int player, float epsilon0, float epsilonGlobal, float epsilonLocal,
                                                         int globalStrategy, int maxDepth, int nodeID, double evaluationBound,
                                                         int maxAllowedInactions) throws Exception {

        /* ************************************************************************************************************
         * Phase 1 : For each unit, rank the unitActions according to preference. Ranking is done by way of calculating
         * distributions.
         *
         *********************************************************************************************************** */
        List<double []> distributions = new LinkedList<>();
        List<Integer> toSample = new LinkedList<>();

        // Calculate the distribution of unit actions for each unit.
        for (UnitActionsTableElement element : unitActionsTable) {

            // for each unit action, compute the relevant distribution
            double [] distribution = new double[element.actionCount];
            int bestIndex = -1;
            double bestEvaluation = 0;
            int lastVisitsCount = 0;

            for (int i = 0 ; i < element.actionCount; i++) {
                /* Three cases can be identified. (1) The initial case, where bestIndex is -1, the conditional test is
                 * passed and the bestIndex receives the index of the first element. In case, the first element was visited
                 * the bestEvaluation will be calculated and the lastVisitsCount will be saved. (2) If the last visit count
                 * is non zero and the current action is unvisited, then the bestIndex is given the current index and
                 * LastVisitsCount and bestEvaluation will be reduced to zero, and the final value of both will remain
                 * zero. (3) If all unit actions have a visit count superior than zero, then the bestIndex will receive
                 * the index of the unit action with the highest reward value. In all cases, the final value of the elements
                 * of the distribution array will be equal to epsilonLocal / element.actionsCount.
                 */
                if (type == 0) { // Max node
                    if (bestIndex == -1 || // (1) initial case.
                       (lastVisitsCount != 0 && element.visitsCount[i] == 0) || // (2) at least one unvisited unit action.
                       (lastVisitsCount != 0 && (element.accumulatedEvaluation[i] / element.visitsCount[i]) > bestEvaluation)) {
                        bestIndex = i;
                        if (element.visitsCount[i] > 0)
                            bestEvaluation = (element.accumulatedEvaluation[i] / element.visitsCount[i]);
                        else
                            bestEvaluation = 0;
                        lastVisitsCount = element.visitsCount[i];
                    }
                } else { // Min node
                    if (bestIndex == -1 ||
                       (lastVisitsCount != 0 && element.visitsCount[i] == 0) ||
                       (lastVisitsCount != 0 && (element.accumulatedEvaluation[i] / element.visitsCount[i] < bestEvaluation))) {
                       bestIndex = i;
                       if (element.visitsCount[i] > 0)
                           bestEvaluation = (element.accumulatedEvaluation[i] / element.visitsCount[i]);
                       else
                           bestEvaluation = 0;
                       lastVisitsCount = element.visitsCount[i];
                    }
                }
                // Distribution value is constant for all unit actions. Will be modified later.
                distribution[i] = epsilonLocal / element.actionCount;
            }

            // Amplify the distribution of the best unit action in case all unit actions were visited (case (3))
            if (element.visitsCount[bestIndex] != 0)
                distribution[bestIndex] = (1 - epsilonLocal) + (epsilonLocal / element.actionCount);
            else {
            // In the other case, (2), and if exploreNonSampledActions is activated, every visited action relevant
            // distribution value will be rendered zero, in order to allow a higher exploration chance for unvisited actions.
                if (exploreNonSampledActions) {
                    for (int i = 0; i < distribution.length; i++)
                        if (element.visitsCount[i] > 0) distribution[i] = 0;
                }
            }

            // Add the items to the relevant list.
            toSample.add(distributions.size()); // Adds the index of the current distribution.
            distributions.add(distribution); // Add the distribution to the distributions list.
        }

        /* *****************************************************************************************************
         * Phase 2 : Select the best combination that results in a valid playerAction by epsilon greedy sampling.
         * Compute the resource usage of the unit actions in the current game state.
         * *****************************************************************************************************/
        ResourceUsage currentResourceUsage = new ResourceUsage();
        for (Unit unit : gameState.getUnits()) {
            UnitAction unitAction = gameState.getUnitAction(unit);
            if (unitAction != null) {
                ResourceUsage resourceUsage = unitAction.resourceUsage(unit, gameState.getPhysicalGameState());
                currentResourceUsage.merge(resourceUsage);
            }
        }

        PlayerAction playerAction = new PlayerAction(); // The player action to construct.
        playerAction.setResourceUsage(currentResourceUsage.clone());
        BigInteger playerActionCode = BigInteger.ZERO; // The player action identifier.

        while (!toSample.isEmpty()) { // Loop through the units.

            // Unit, Unit-Action level
            // Remove a unit index at random, and choose an action for it.
            int unitIndex = toSample.remove(random.nextInt(toSample.size()));

            try {
                // Search for a unit action. Get relevant table element and distribution array.
                UnitActionsTableElement element = unitActionsTable.get(unitIndex);
                double [] distribution = distributions.get(unitIndex);

                List<Double> distributionList = new ArrayList<>(); // distribution array converted to list.
                List<Integer> distributionOutputs = new ArrayList<>(); // indices of actions and distribution items.

                for (int i = 0; i < distribution.length; i++) {
                    distributionList.add(distribution[i]);
                    distributionOutputs.add(i);
                }

                // Try at random
                int unitActionCode = (Integer) Sampler.weighted(distributionList, distributionOutputs);
                UnitAction unitAction = element.actions.get(unitActionCode);
                ResourceUsage unitActionResourceUsage;

                int noneActionCode = element.actionCount - 1;
                BigInteger rejectedActionCode;
//                System.out.println(element.actions.get(noneActionCode));

                // Filter out idle unit actions **************** (Pos 1)
                if (distributionList.size() > 1 && unitAction.getType() == UnitAction.TYPE_NONE &&
                    maxAllowedInactions != -1) {
                    rejectedActionCode = BigInteger.valueOf(noneActionCode).multiply(multipliers[unitIndex]);
                    boolean previouslyRejected = rejectedActions.contains(rejectedActionCode);

                    if (inactionsCounter >= maxAllowedInactions || previouslyRejected) {
                        // Remove the unit action and sample another one.
//                        System.out.println(this.nodeID + " Inaction Pruned");
                        int noneActionIndex = distributionOutputs.indexOf(noneActionCode);
                        distributionList.remove(noneActionIndex);
                        distributionOutputs.remove(noneActionIndex);
                        /*if (!previouslyRejected)*/
                        rejectedActions.add(rejectedActionCode);

                        unitActionCode = (Integer) Sampler.weighted(distributionList, distributionOutputs);
                        unitAction = element.actions.get(unitActionCode);
                    } else {
//                        System.out.println(this.nodeID + " Inaction Allowed");
                        inactionsCounter++;
                    }
                    /*previouslyRejected = false;*/
                }

                unitActionResourceUsage = unitAction.resourceUsage(element.unit, gameState.getPhysicalGameState());
                // Loop through the individual unit actions.
                // Check consistency of the unit action with the playerAction.
                while (!playerAction.getResourceUsage().consistentWith(unitActionResourceUsage, gameState)) {
                    // In case the unit action sampled not being consistent with the base resource usage.
                    // Find the index of the inconsistent unit action and remove it from distributionList and outputs.
                    int unitActionIndex = distributionOutputs.indexOf(unitActionCode);
                    distributionList.remove(unitActionIndex);
                    distributionOutputs.remove(unitActionIndex);

                    // Try another unit action.
                    if (!distributionList.isEmpty()) {
                        unitActionCode = (Integer) Sampler.weighted(distributionList, distributionOutputs);
                        unitAction = element.actions.get(unitActionCode);
                    } else {
                        unitActionCode = noneActionCode;
                        unitAction = new UnitAction(UnitAction.TYPE_NONE, waitDuration);
                    }

                    // Filter out idle actions *************** (Pos 2)
                    if (distributionList.size() > 1 && unitAction.getType() == UnitAction.TYPE_NONE
                            && maxAllowedInactions != -1) {
                        rejectedActionCode = BigInteger.valueOf(noneActionCode).multiply(multipliers[unitIndex]);
                        boolean previouslyRejected = rejectedActions.contains(rejectedActionCode);
                        /*rejectedActionCode = BigInteger.valueOf(noneActionCode).multiply(multipliers[unitIndex]);
                        if (!rejectedActions.contains(rejectedActionCode))
                            rejectedActions.add(rejectedActionCode);
                        else
                            previouslyRejected = true;*/

                        if (inactionsCounter >= maxAllowedInactions || previouslyRejected) {
                            // Remove the unit action and sample another one.
                            int noneActionIndex = distributionOutputs.indexOf(noneActionCode);
                            distributionList.remove(noneActionIndex);
                            distributionOutputs.remove(noneActionIndex);
                            /*if (!previouslyRejected)*/
                            rejectedActions.add(rejectedActionCode);

                            unitActionCode = (Integer) Sampler.weighted(distributionList, distributionOutputs);
                            unitAction = element.actions.get(unitActionCode);
                        } else
                            inactionsCounter++;
                    }

                    unitActionResourceUsage = unitAction.resourceUsage(element.unit, gameState.getPhysicalGameState());
                }

                /*if (distributionList.isEmpty())
                    unitAction = new UnitAction(UnitAction.TYPE_NONE, 10);*/

                // Debug:
                if (gameState.getUnit(element.unit.getID()) == null) throw new Error("Issuing an action to an inexistant unit");

                // At this point a consistent unit action is found.
                playerAction.getResourceUsage().merge(unitActionResourceUsage);
                playerAction.addUnitAction(element.unit, unitAction);

                // Compute the player action code incrementally.
                playerActionCode = playerActionCode.add(BigInteger.valueOf(unitActionCode).multiply(multipliers[unitIndex]));

            } catch (Exception exception) {
                exception.printStackTrace();
            }
        }

//        System.out.println(playerAction);

        // Check whether a node of the same playerActionCode already exists.
        NMCTSRIPFWaitDurationNode oldChild = childrenMap.get(playerActionCode);
        if (oldChild == null) { // If no node with the same playerActionCode exists, create one.
            actions.add(playerAction);
            GameState newGameState = gameState.cloneIssue(playerAction);
            NMCTSRIPFWaitDurationNode newChild = new NMCTSRIPFWaitDurationNode(player, newGameState.clone(), this, nodeID, exploreNonSampledActions, waitDuration);
            childrenMap.put(playerActionCode, newChild);
            children.add(newChild);
            return newChild;
        }

        // If a child with the same playerActionCode exists, go down the tree, i.e. select a child from his children.
        return oldChild.selectLeaf(player, epsilon0, epsilonGlobal, epsilonLocal, globalStrategy, maxDepth, nodeID, evaluationBound, maxAllowedInactions);
    }

    /**
     * Backpropagate the simulation's results up the tree nodes, updating the respective unit action tables.
     * @param evaluation The result of the simulation.
     * @param child
     */
    public void backpropagate(double evaluation, NMCTSRIPFWaitDurationNode child) {
        // Update the node's accumulated evaluation and visit count.
        accumulatedEvaluation += evaluation;
        visitCount++;

        if (child != null) { // if a child is provided.
            // Extract the index of the child and the relevant playerAction.
            int index = children.indexOf(child);
            PlayerAction playerAction = actions.get(index);

            // For each unitAction in the playerAction, update the respective accumulated evaluation and visit count.
            for (Pair<Unit, UnitAction> unitAction : playerAction.getActions()) {
                // Extract the unit's relevant unitActionsTable element.
                UnitActionsTableElement element = getActionTableElement(unitAction.m_a);
                index = element.actions.indexOf(unitAction.m_b); // Extract the index of the chosen unitAction

                if (index == -1) { // in case the unit action does not exist.
                    System.out.println("Looking for action : " + unitAction.m_b);
                    System.out.println("Available actions are : " + element.actions);
                }

                // Update the statistics of the chosen unit action in its relevant position in the accumulatedEvaluation
                // and visitsCount array of the unit's element.
                element.accumulatedEvaluation[index] += evaluation;
                element.visitsCount[index]++;
            }
        }

        // If the node has a parent, backpropagate to this parent, sending a reference to this child as an argument.
        if (parent != null)
            parent.backpropagate(evaluation, this);
    }

    /**
     * Returns the action table element for the given unit.
     * @param unit The unit in question.
     * @return The relevant unit action table element.
     */
    private UnitActionsTableElement getActionTableElement(Unit unit) {
        for (UnitActionsTableElement element : unitActionsTable)
            if (element.unit == unit) return element;
        throw new Error("Could not find Action Table element.");
    }

    public CustomPlayerActionGenerator getActionGenerator() {
        return actionGenerator;
    }

    public GameState getGameState() {
        return gameState;
    }

    public List<PlayerAction> getActions() {
        return actions;
    }

    public List<NMCTSRIPFWaitDurationNode> getChildren() {
        return children;
    }

    public int getVisitCount() {
        return visitCount;
    }

    public double getAccumulatedEvaluation() {
        return accumulatedEvaluation;
    }
}
