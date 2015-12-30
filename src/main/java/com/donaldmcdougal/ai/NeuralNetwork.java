/**
 * 
 */
package com.donaldmcdougal.ai;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.MLResettable;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.strategy.SmartLearningRate;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.util.obj.ObjectCloner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.Banner;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Runs the neural network.
 * @author donaldmcdougal
 *
 */
@SpringBootApplication
public class NeuralNetwork implements CommandLineRunner {

	private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);
	
	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(NeuralNetwork.class);
		app.setBannerMode(Banner.Mode.OFF);
		app.run(args);
	}
	
	/* (non-Javadoc)
	 * @see org.springframework.boot.CommandLineRunner#run(java.lang.String[])
	 */
	@Override
	public void run(String... args) throws Exception {
		
		//log.debug("Running Feed Forward Network");
		//this.runFeedForwardNetwork();
		
		log.debug("Running Feed Forward Genetic Network");
		this.runFeedForwardGeneticNetwork();
		
		//log.debug("Running Elman Network");
		//this.runElmanNetwork();
		
		//log.debug("Running Genetic Elman Network");
		//this.runGeneticElmanNetwork();
		
		Encog.getInstance().shutdown();
	}
	
	private void runFeedForwardNetwork() {
		double XOR_INPUT[][] = { { 0.0 , 0.0 } , { 1.0 , 0.0 } , { 0.0 , 1.0 } , { 1.0 , 1.0 } };
		double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } }; 
		
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		
		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		
		int epoch = 1;
		do {
			train.iteration();
			log.debug("Epoch # {}, Error {}", epoch, train.getError());
			epoch++;
		} while (train.getError() > 0.0001);
		train.finishTraining();
		
		log.debug("Neural network results");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			log.debug("input {} and {}, actual = {}, ideal = {}", pair.getInput().getData(0), pair.getInput().getData(1), output.getData(0), pair.getIdeal().getData(0));
		}
		
		log.debug("Error rate: {}", train.getError() * 100 + "%");
	}
	
	private void runFeedForwardGeneticNetwork() {
		double XOR_INPUT[][] = { { 0.0 , 0.0 } , { 1.0 , 0.0 } , { 0.0 , 1.0 } , { 1.0 , 1.0 } };
		double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } }; 
		
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain trainAlt = new NeuralSimulatedAnnealing(network, score, 10, 2, 100);
		MLTrain trainMain = new MLMethodGeneticAlgorithm(new MethodFactory() {
			@Override
			public MLMethod factor() {
				final MLMethod result = (MLMethod) ObjectCloner.deepCopy(network);
				((MLResettable) result).reset();
				return result;
			}
		}, score, 1000);
		
		StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);
		
		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			log.debug("Training {}, Epoch {}, Error: {}", "Feed Forward", epoch, trainMain.getError());
			epoch++;
		}
		
		log.debug("Neural network results");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			log.debug("input {} and {}, actual = {}, ideal = {}", pair.getInput().getData(0), pair.getInput().getData(1), output.getData(0), pair.getIdeal().getData(0));
		}
		
		log.debug("Error rate: {}", trainMain.getError() * 100 + "%");
	}
	
	private void runElmanNetwork() {
		MLDataSet trainingSet = this.generate(120);

		BasicNetwork elmanNetwork = this.createElmanNetwork();

		double elmanError = this.trainElmanNetwork("Elman", elmanNetwork, trainingSet);
		
		for (MLDataPair pair : trainingSet) {
			MLData output = elmanNetwork.compute(pair.getInput());
			log.debug("input {}, actual = {}, ideal = {}", pair.getInput().getData(0), output.getData(0), pair.getIdeal().getData(0));
		}

		System.out.println("Best error rate with Elman Network: " + elmanError);
	}
	
	private void runGeneticElmanNetwork() {
		MLDataSet trainingSet = this.generate(120);
		BasicNetwork elmanNetwork = this.createElmanNetwork();
		double elmanError = this.trainGeneticElmanNetwork("Genetic Elman", elmanNetwork, trainingSet);
		for (MLDataPair pair : trainingSet) {
			MLData output = elmanNetwork.compute(pair.getInput());
			log.debug("input {}, actual = {}, ideal = {}", pair.getInput().getData(0), output.getData(0), pair.getIdeal().getData(0));
		}
		System.out.println("Best error rate with Genetic Elman Network: " + elmanError);
	}
	
	private BasicNetwork createElmanNetwork() {
		// construct an Elman type network
		ElmanPattern pattern = new ElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(1);
		pattern.addHiddenLayer(6);
		pattern.setOutputNeurons(1);
		return (BasicNetwork)pattern.generate();
	}
	
	private double trainElmanNetwork(String what, BasicNetwork network, MLDataSet trainingSet) {
		// train the neural network
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain trainAlt = new NeuralSimulatedAnnealing(network, score, 10, 2, 1000);
		MLTrain trainMain = new Backpropagation(network, trainingSet, 0.000001, 0.0);

		StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new SmartLearningRate());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			log.debug("Training {}, Epoch {}, Error: {}", what, epoch, trainMain.getError());
			epoch++;
		}
		return trainMain.getError();
	}
	
	private double trainGeneticElmanNetwork(String what, BasicNetwork network, MLDataSet trainingSet) {
		// train the neural network
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain trainAlt = new NeuralSimulatedAnnealing(network, score, 10, 2, 100);
		MLTrain trainMain = new MLMethodGeneticAlgorithm(new MethodFactory() {
			@Override
			public MLMethod factor() {
				final MLMethod result = (MLMethod) ObjectCloner.deepCopy(network);
				((MLResettable) result).reset();
				return result;
			}
		}, score, 1000);
		
		StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new ResetStrategy(.04, 1000));
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			log.debug("Training {}, Epoch {}, Error: {}", what, epoch, trainMain.getError());
			epoch++;
		}
		return trainMain.getError();
	}
	
	private MLDataSet generate(int count) {
		/**
		 * 1 xor 0 = 1, 0 xor 0 = 0, 0 xor 1 = 1, 1 xor 1 = 0
		 */
		double[] sequence = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
				0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };
		
		double[][] input = new double[count][1];
		double[][] ideal = new double[count][1];

		for (int i = 0; i < count; i++) {
			input[i][0] = sequence[i % sequence.length];
			ideal[i][0] = sequence[(i + 1) % sequence.length];
		}

		return new BasicMLDataSet(input, ideal);
	}
}
