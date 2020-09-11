package ai.mcts.naivemcts;

import aes.nmcts.NMCTSNode;
import aes.nmcts.NMCTSRandomInactivityFilteringProbaNode;
import ai.RandomBiasedAI;
import ai.RandomBiasedWaitDurationAI;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.InterruptibleAI;
import ai.core.ParameterSpecification;
import ai.evaluation.EvaluationFunction;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

import java.util.ArrayList;
import java.util.List;

public class NMCTSRIPRWaitDuration extends AIWithComputationBudget implements InterruptibleAI {

    // Members ************************************************************************************
    private GameState initialGameState;
    private EvaluationFunction evaluationFunction;
    private AI playoutPolicy = new RandomBiasedAI();
    private NMCTSRIPRWaitDurationNode tree;

    private int player;
    private int simulationTime = 1024;
    private int maxDepth = 10;
    private long maxActions = 0;
    private int currentIteration = 0;

    // NaïveMCTS Specific attributes. The epsilon-greedy strategies parameters.
    private float epsilon0 = 0.2f;
    private float epsilonGlobal = 0.0f;
    private float epsilonLocal = 0.25f;

    private int globalStrategy = NMCTSNode.EPSILON_GREEDY;
    private boolean exploreNonSampledActions = true;
    private double evaluationBound;

    // Stats
    long totalRuns = 0;
    long totalCyclesExecuted = 0;
    long totalActionsIssued = 0;
    long totalTime = 0;

    // Inactivity Filtering
    float allowProbability = 0.0f;

    int waitDuration = 10;
    int playoutWaitDuration = 10;

    public NMCTSRIPRWaitDuration(UnitTypeTable unitTypeTable, float allowProbability, int waitDuration, int playoutWaitDuration) {
        this(100, -1, 100, 10, 0.4f, 0.0f, 0.3f,
                NMCTSRIPRWaitDurationNode.EPSILON_GREEDY, new RandomBiasedAI(),
                new SimpleSqrtEvaluationFunction3(), true,allowProbability, waitDuration, playoutWaitDuration);
    }

    /**
     *
     * @param unitTypeTable
     */
    public NMCTSRIPRWaitDuration(UnitTypeTable unitTypeTable) {
        this(100, -1, 100, 10, 0.4f, 0.0f, 0.3f,
                NMCTSRIPRWaitDurationNode.EPSILON_GREEDY, new RandomBiasedAI(),
                new SimpleSqrtEvaluationFunction3(), true,0f,10,10);
    }

    /**
     * Constructs the controller with the specified time and iterations budget
     *
     * @param timeBudget       time in milliseconds
     * @param iterationsBudget number of allowed iterations
     */
    public NMCTSRIPRWaitDuration(int timeBudget, int iterationsBudget, int simulationTime, int maxDepth,
                                 float epsilon0, float epsilonGlobal, float epsilonLocal, int globalStrategy, AI playoutPolicy,
                                 EvaluationFunction evaluationFunction, boolean exploreNonSampledActions,
                                 float allowProbability, int waitDuration, int playoutWaitDuration) {
        super(timeBudget, iterationsBudget);
        this.simulationTime = simulationTime;
        this.maxDepth = maxDepth;
        this.epsilon0 = epsilon0;
        this.epsilonGlobal = epsilonGlobal;
        this.epsilonLocal = epsilonLocal;
        this.globalStrategy = globalStrategy;
        this.playoutPolicy = playoutPolicy;
        this.evaluationFunction = evaluationFunction;
        this.exploreNonSampledActions = exploreNonSampledActions;

        this.allowProbability = allowProbability;
        this.waitDuration = waitDuration;
        this.playoutWaitDuration = playoutWaitDuration;
        if (playoutWaitDuration != 10) this.playoutPolicy = new RandomBiasedWaitDurationAI(playoutWaitDuration);
    }

    /**
     *
     * @param player ID of the player to move. Use it to check whether units are yours or enemy's
     * @param gameState
     * @return
     * @throws Exception
     */
    @Override
    public PlayerAction getAction(int player, GameState gameState) throws Exception {
        if (gameState.canExecuteAnyAction(player)) {
            startNewComputation(player, gameState.clone());
            computeDuringOneGameFrame();
            return getBestActionSoFar();
        } else
            return new PlayerAction();
    }

    /**
     * Initiates the computation of the optimal action to return.
     * @param player the index of the player have its action calculated
     * @param gameState
     * @throws Exception
     */
    @Override
    public void startNewComputation(int player, GameState gameState) throws Exception {
        this.player = player;
        currentIteration = 0;
        initialGameState = gameState;
        // Create the search tree, and increase the currentIteration counter afterwards.
        tree = new NMCTSRIPRWaitDurationNode(player, gameState, null, currentIteration++, exploreNonSampledActions, waitDuration);
        evaluationBound = evaluationFunction.upperBound(gameState);

        if (tree.getActionGenerator() == null)
            maxActions = 0;
        else
            maxActions = Math.max(tree.getActionGenerator().getSize(), maxActions);
    }

    @Override
    public void computeDuringOneGameFrame() throws Exception {
        // Time budget
        long startTime = System.currentTimeMillis();
        long endTime = startTime;
        // Iteration budget
        long iterations = 0;

        // Main loop
        while (true) {
            if (!monteCarloRun(player)) break;
            iterations++;
            endTime = System.currentTimeMillis();
            if (TIME_BUDGET >= 0 && (endTime - startTime) >= TIME_BUDGET) break;
            if (ITERATIONS_BUDGET >= 0 && iterations >= ITERATIONS_BUDGET) break;
        }

        // Stats
        totalTime += (endTime - startTime);
        totalCyclesExecuted++;
    }

    private boolean monteCarloRun(int player) throws Exception {
        // (1) Selection and Expansion
        NMCTSRIPRWaitDurationNode selected = tree.selectLeaf(player, epsilon0, epsilonGlobal, epsilonLocal,
                globalStrategy, maxDepth, currentIteration++, evaluationBound, allowProbability);

        if (selected != null) {
            // (2) Simulation and evaluation
            GameState simGameState = selected.getGameState().clone();
            simulate(simGameState, simGameState.getTime() + simulationTime);

            int time = simGameState.getTime() - initialGameState.getTime();
            double evaluation = evaluationFunction.evaluate(player, 1 - player, simGameState);
            evaluation *= Math.pow(0.99, time / 10.0);

            // (3) Backpropagation
            selected.backpropagate(evaluation, null);
            totalRuns++;
        } else {
            System.err.println(this.getClass().getSimpleName() + " : Claims there are no more leafs to explore !");
            return false;
        }
        return true;
    }

    @Override
    public PlayerAction getBestActionSoFar() throws Exception {
        int bestActionIndex = getMostVisitedActionIndex();
        if (bestActionIndex == -1)
            return new PlayerAction();
        return tree.getActions().get(bestActionIndex);
    }

    private int getMostVisitedActionIndex() {
        totalActionsIssued++;

        NMCTSRIPRWaitDurationNode mostVisitedChild = null; // The most visited child
        int mostVisitedChildIndex = -1; // The most visited child index

        if (tree.getChildren() == null)
            return -1;

        // Find the most visited child.
        for (int index = 0; index < tree.getChildren().size(); index++) {
            NMCTSRIPRWaitDurationNode child = tree.getChildren().get(index);
            if (mostVisitedChild == null || child.getVisitCount() > mostVisitedChild.getVisitCount()) {
                mostVisitedChild = child;
                mostVisitedChildIndex = index;
            }
        }

        return mostVisitedChildIndex;
    }

    private int getHighestEvaluatedActionIndex() {
        totalActionsIssued++;

        NMCTSRIPRWaitDurationNode highestEvaluatedChild = null; // The highest evaluated child.
        int highestEvaluatedChildIndex = -1; // The highest evaluated child index.

        if (tree.getChildren() == null)
            return -1;

        // Find the highest evaluated child.
        for (int index = 0; index < tree.getChildren().size(); index++) {
            NMCTSRIPRWaitDurationNode child = tree.getChildren().get(index);
            if (highestEvaluatedChild == null ||
               (child.getAccumulatedEvaluation() / (double) child.getVisitCount()) >
                       (highestEvaluatedChild.getAccumulatedEvaluation() / (double) highestEvaluatedChild.getVisitCount())) {
                highestEvaluatedChild = child;
                highestEvaluatedChildIndex = index;
            }
        }

        return highestEvaluatedChildIndex;
    }

    private void simulate(GameState gameState, int simTime) throws Exception {
        boolean gameOver = false;

        do {
            if (gameState.isComplete())
                gameOver = gameState.cycle();
            else {
                gameState.issue(playoutPolicy.getAction(0, gameState));
                gameState.issue(playoutPolicy.getAction(1, gameState));
            }
        } while (!gameOver && gameState.getTime() < simTime);
    }

    @Override
    public void reset() {
        tree = null;
        initialGameState = null;
        currentIteration = 0;
        // Reset stats.
        totalRuns = 0;
        totalCyclesExecuted = 0;
        totalActionsIssued = 0;
        totalTime = 0;
    }

    public void resetSearch() {
        tree = null;
        initialGameState = null;
    }

    @Override
    public AI clone() {
        return new NMCTSRIPRWaitDuration(TIME_BUDGET, ITERATIONS_BUDGET, simulationTime, maxDepth,
                         epsilon0, epsilonGlobal, epsilonLocal, globalStrategy, playoutPolicy,
                         evaluationFunction, exploreNonSampledActions, allowProbability, waitDuration, playoutWaitDuration);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + allowProbability + ", " + waitDuration + ", " + playoutWaitDuration + ")";

    }

    @Override
    public String statisticsString() {
        return "Total Runs: " + totalRuns +
               ", Runs per action: " + (totalRuns / (float) totalActionsIssued) +
               ", Runs per cycle: " + (totalRuns / (float) totalCyclesExecuted) +
               ", Average time per cycle: " + (totalTime / (float) totalCyclesExecuted) +
               ", Max branching factor: " + maxActions;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();

        parameters.add(new ParameterSpecification("TimeBudget", int.class, 100));
        parameters.add(new ParameterSpecification("IterationsBudget", int.class, -1));
        parameters.add(new ParameterSpecification("SimulationTime", int.class, 100));
        parameters.add(new ParameterSpecification("MaxDepth", int.class, 10));

        parameters.add(new ParameterSpecification("Epsilon0", float.class, 0.0f));
        parameters.add(new ParameterSpecification("EpsilonGlobal", float.class, 0.0f));
        parameters.add(new ParameterSpecification("EpsilonLocal", float.class, 0.0f));

        parameters.add(new ParameterSpecification("PlayoutPolicy", AI.class, playoutPolicy));
        parameters.add(new ParameterSpecification("EvaluationFunction", EvaluationFunction.class, new SimpleSqrtEvaluationFunction3()));
        parameters.add(new ParameterSpecification("ExploreNonSampledActions", boolean.class, true));

        return parameters;
    }

    public float getAllowProbability() {
        return allowProbability;
    }

    public void setAllowProbability(float allowProbability) {
        this.allowProbability = allowProbability;
    }
}
