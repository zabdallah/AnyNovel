package conceptEvolution;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.PriorityQueue;

import StreamAR.BLM;
import StreamAR.NovelPredection;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.TaskLogger;

public class AnyNovelLauncher {
	static String m_History = "";
	static StringBuffer outBuff = new StringBuffer();
	private static Instances m_Instances;
	static boolean validArg = false;
	static boolean validPhase = false;
	static String sdet = "";
	static String strTP = "";
	static String strTN = "";
	static String strFP = "";
	static String strFN = "";
	static String strFA = "";
	static String strTR = "";
	static String strUN = "";
	static int fa_counter = 0;
	static int fa_ptsCounter = 0;
	static int rec_counter = 0;
	static int rec_ptscounter = 0;
	static String s = "\n";
	double confusionMatrix[][]; 

	public static void run(BLM model, String test_dataset, String valid_dataset) {
		try {
			// Validation phase if -valid argument provided
			if (!valid_dataset.isEmpty()) {
				System.out.println("Validation Phase:\n---------\n");
				validArg = true;
				validPhase = true;
				model = runOnData(model, valid_dataset);
				System.out.println(model.modelStatistices());
			}
			// Testing phase
			validPhase = false;
			runOnData(model, test_dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Throwable e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private static BLM runOnData(BLM model, String dataset) throws Throwable {
		String filePath = new File("").getAbsolutePath();
		filePath = filePath.concat("/Data/Test/" + dataset);
		File testFile = new File(filePath);
		System.out.println(testFile);
		try {
			DataSource source = new DataSource(testFile.getAbsolutePath());
			m_Instances = source.getDataSet();

		} catch (Exception e) {
			System.out.println(
					"Problem in the file name/path, file name with extension is required. File must be in /Test/");
			e.printStackTrace();
		}

		// setting class attribute if the data format does not provide this
		// information
		// For example, the ARFF format saves the class attribute information as
		// well
		if (m_Instances.classIndex() == -1)
			m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
		long trainTimeStart = 0;
		long trainTimeElapsed = 0;
		Instances m_testInstaces = new Instances(m_Instances);
		m_testInstaces.setClassIndex(m_testInstaces.numAttributes() - 1);
		trainTimeStart = System.currentTimeMillis();
		// Call anyNovel on Test data, parameters and using Model
		BLM newModel = AnyNovel(model, m_testInstaces, readParameters(dataset, model));
		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
		System.out.println("Time Elapsed: " + trainTimeElapsed);
		// System.out.println(outBuff);
		return newModel;
	}

	private static HashMap<String, String> readParameters(String dataset, BLM model) throws IOException {
		HashMap<String, String> parameters = new HashMap<String, String>();
		String filepath = new File("").getAbsolutePath();
		File fc = new File(filepath.concat("/parameters.txt"));
		BufferedReader in = new BufferedReader(new FileReader(fc.getPath()));
		String line;
		while ((line = in.readLine()) != null) {
			String p[] = line.trim().split("=");
			parameters.put(p[0].trim(), p[1].trim());
			// System.out.println(p[0] +" "+p[1]);

		}

		return parameters;
	}

	private static BLM AnyNovel(BLM BaseModel, Instances StreamInst, HashMap<String, String> parameters)
			throws Throwable {
		// Implement AnyNovel and returning results
		// Thread m_RunThread= null;
		//
		// m_RunThread = new Thread() {
		// Later: # Thread??!
		// Build the model

		// Variables:
		String fileOutStr = "";
		int counter_w = 0;

		ArrayList<NovelPredection> NovelStats = new ArrayList<NovelPredection>();
		// CEModel: starts from BLM and expand it for novel concepts
		BLM CEModel = new BLM(BaseModel);
		CEModel.resetAll();
		// Deactivate validation for now
		boolean validFlag = false;
		// Step1: Set the parameters

		CEModel.setParameters(parameters);

		
		// Set test data
		CEModel.setTestInstancesHeader(StreamInst);

		int segSize = Integer.parseInt(parameters.get("Segment_Size"));

		// Step 2: Set the model and print stats about the baseline model
		String logName = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
		System.out.println("-----------------------------------");
		System.out.println(logName + " Concept Evolution");
		System.out.println("Setting up...");

		// Segmentation: Split the stream into equal size chunks

		System.out.println("Started segmentation");

		int n = StreamInst.numInstances() / segSize;
		Instances inst = null;
		// Move to the end// after the segmentation loop
		int unknown = 0;
		int unknownC = 0;

		confusionMatrix[][] = new double[CEModel.getOldLabels().length + 1][CEModel.getOldLabels().length + 2];
		// Update only for this dataset>
		// Not accumulated update.

		double avPurity = 0;
		int pCount = 0;
		String updateStr = "";
		boolean lastSeg = false;
		ArrayList<String> detARR = new ArrayList<String>();
		ArrayList<Integer> updatedCounter = new ArrayList<Integer>();
		ArrayList<Integer> activeCounter = new ArrayList<Integer>();
		fileOutStr += " ID,Case, size of true label|total,maj label,Distance to existing classes,Draft_flag, buffer, INSIDE, INSLACK, hitInsideID, InSlackID, BufferSize,Draft_flag, buffer, dcision,Movement,Stability,,, \n";
		for (int i = 0; i < n + 1; i++) {
			if (i == n) {
				// End of stream, when remaining instances size is less than seg
				// size
				int remain = StreamInst.numInstances() - (n * segSize);
				if (remain != 0) {
					inst = new Instances(StreamInst, n * segSize, remain);
					lastSeg = true;
				} else
					break;
			} else {
				int index = i * segSize;
				inst = new Instances(StreamInst, index, segSize);
			}
			inst.setClassIndex(inst.numAttributes() - 1);

			Instances existInstances = new Instances(inst);
			String[] attribNames = new String[StreamInst.numAttributes()];
			existInstances.setClassIndex(attribNames.length - 1);
			int xclass = existInstances.classAttribute().numValues();
			// Cluster each segment into 2 clusters to capture interleaved and
			// overlapped concepts
			String[] options = new String[2];
			options[0] = "-N"; // number of clusters
			options[1] = "2";
			EM clusterer = new EM();
			clusterer.setOptions(options); // set the options
			clusterer.buildClusterer(removeClass(existInstances));

			String[] classValues = new String[xclass];
			int f = 0;
			for (Enumeration<?> c = existInstances.classAttribute().enumerateValues(); c.hasMoreElements();) {
				classValues[f] = (String) c.nextElement();
				f++;
			}
			ClusterEvaluation eval2 = new ClusterEvaluation();
			eval2.setClusterer(clusterer);
			eval2.evaluateClusterer(existInstances, "", false);

			// Build test model

			Instances trainNolabel = removeClass(existInstances);
			double[] Assignments = eval2.getClusterAssignments();
			Instances[] AssignmentArr = null;

			AssignmentArr = new Instances[clusterer.numberOfClusters()];

			// Initialize
			for (int j = 0; j < clusterer.numberOfClusters(); j++) {
				AssignmentArr[j] = new Instances(existInstances, existInstances.numInstances());
			}
			// Fill in assignmentArr with clusters members
			for (int count = 0; count < trainNolabel.numInstances(); count++) {
				Instance tmpInst = (Instance) existInstances.instance(count).copy();
				AssignmentArr[(int) Assignments[count]].add(tmpInst);
			}

			for (int p = 0; p < AssignmentArr.length; p++)
				AssignmentArr[p].compactify();

			// Process each cluster (Each element in Assignment ARR):
			for (int p = 0; p < AssignmentArr.length; p++) {
			
				sdet = "";
				strTP = "";
				strTN = "";
				strFP = "";
				strFN = "";
				strFA = "";
				strTR = "";
				strUN = "";
				fa_counter = 0;
				fa_ptsCounter = 0;
				rec_counter = 0;
				rec_ptscounter = 0;
				if (AssignmentArr[p].numInstances() != 0) {
					boolean declared = false;
					String NP = CEModel.adaptationComponent(AssignmentArr[p]);

					if (NP.contains("Case 0.1") || NP.contains("Case 0.0") || NP.contains("Case 0.5")) {
						if (i == n)
							declared = true;
						NovelPredection pred = CEModel.noveltyStatistics();
						updateResults(NP, pred);
						
						//
						if (validFlag) {

							s += CEModel.updateModelConceptEvolution(pred);
							updateStr += i + ",";

							if (s.contains("No-Update") || s.contains("No-Change")) {
								updateStr += "0 \n";
								activeCounter.add(-1);
							} else if (s.contains("Update-Model")) {
								updateStr += "1 \n";
								updatedCounter.add(i);
							} else
								updateStr += "-1 \n";

						}
						outBuff.append(pred.getDetails());
						outBuff.append(s);
						NP = CEModel.adaptationComponent(AssignmentArr[p]);

					}
					outBuff.append(i + ",");
					outBuff.append(NP);
					// Decision cases

					if (NP.contains("Case 1.1") || NP.contains("Case 2.1") || NP.contains("Case 3.2")
							|| NP.contains("Case 2.2") || NP.contains("Case 3.1") || NP.contains("Case 4.3")
							|| NP.contains("Case 2.3") || NP.contains("Case 3.4") || NP.contains("Case 4.1")) {
						if (i == n)
							declared = true;
						NovelPredection pred = CEModel.noveltyStatistics();

						s = "*True label= " + pred.getClassTrueLabel()[0] + " Predicted label:( "
								+ pred.getPredictedLabel() + " ) FA_novel= " + pred.IsFAnovel() + " Recurrent Novel= "
								+ pred.isRecurrent() + "  The Noelty Type: " + pred.getNoveltyType() + " Size: "
								+ pred.getClassSize();
						// s+="\n Density: "+
						// pred.getDenNovFlag()+" Gravity: "+
						// pred.getGravNovFlag()+"\n";
						if (pred.getNoveltyType().contains("Tp")) {
							strTP = Integer.toString(pred.getClassSize());
							for (int md = 0; md < CEModel.getNumberOfClasses(); md++) {

								s += " Max Distance= " + CEModel.getMaxDistance(md) + " \n\n";
							}

						}
						double av = Double.parseDouble(pred.getClassTrueLabel()[1]) / pred.getClassSize();
						avPurity += av;
						pCount++;
						if (pred.IsFAnovel() || pred.isRecurrent()) {

							if (pred.IsFAnovel()) {
								strFA = Integer.toString(pred.getClassSize());
								fa_counter++;
								fa_ptsCounter += pred.getClassSize();
							}
							if (pred.isRecurrent()) {
								strTR = Integer.toString(pred.getClassSize());
								rec_counter++;
								rec_ptscounter += pred.getClassSize();

							}
							s += "Novel Class Size: ";
							s += pred.getClassSize();
							s += "  Class majority label: ";
							s += pred.getClassTrueLabel()[0];
							s += "  Purity: " + pred.getClassTrueLabel()[1];
						}

						else if (pred.getNoveltyType().contains("Fn")) {
							strFN = Integer.toString(pred.getClassSize());
							s += "Novel class classified as existing ";

						}
						if (pred.isUnknown()) {
							strUN = Integer.toString(pred.getClassSize());
							unknown += pred.getClassSize();
							unknownC++;
							if (CEModel.getUpateFlag()) {

								s += CEModel.updateModelConceptEvolution_Unknown(CEModel.getDeclaredData(), pred);
								updateStr += i + ",";
							}

						}
						s += "\n";
						if (pred.getNoveltyType().contains("Fn")) {
							confusionMatrix[CEModel.getOldLabels().length][ID(pred.getPredictedLabel(),
									CEModel.getOldLabels())] += pred.getClassSize();
						}
						if (pred.getNoveltyType().contains("Tn")) {
							strTN = Integer.toString(pred.getClassSize());
							confusionMatrix[ID(pred.getClassTrueLabel()[0], CEModel.getOldLabels())][ID(
									pred.getPredictedLabel(), CEModel.getOldLabels())] += pred.getClassSize();
						}
						if (pred.getNoveltyType().contains("Fp")) {
							strFP = Integer.toString(pred.getClassSize());
							confusionMatrix[ID(pred.getClassTrueLabel()[0],
									CEModel.getOldLabels())][CEModel.getOldLabels().length + 1] += pred.getClassSize();
						} else if (pred.getNoveltyType().contains("Tp")) {
							strTP = Integer.toString(pred.getClassSize());
							confusionMatrix[CEModel.getOldLabels().length][CEModel.getOldLabels().length + 1] += pred
									.getClassSize();
						}
						NovelStats.add(pred);
						sdet = segSize + "," + i + "," + p + "," + strTP + "," + strTN + "," + strFP + "," + strFN + ","
								+ strFA + "," + strTR + "," + strUN;
						detARR.add(sdet);

						sdet = "";
						strTP = "";
						strTN = "";
						strFP = "";
						strFN = "";
						strFA = "";
						strTR = "";
						strUN = "";
						if (CEModel.getUpateFlag()) {

							s += CEModel.updateModelConceptEvolution(pred);
							updateStr += i + ",";

							if (s.contains("No-Update") || s.contains("No-Change")) {
								updateStr += "0 \n";
								activeCounter.add(-1);
							} else if (s.contains("Update-Model")) {
								updateStr += "1 \n";
								updatedCounter.add(i);
							} else
								updateStr += "-1 \n";
						}
						outBuff.append(pred.getDetails());
						outBuff.append(s);
					}

					// Very last data
					if (!declared && n == i && p == AssignmentArr.length - 1) {
						CEModel.lastSegDecision();
						NovelPredection pred = CEModel.noveltyStatistics();
						s = "\n *True label= " + pred.getClassTrueLabel()[0] + " Predicted label= "
								+ pred.getPredictedLabel() + "Fa novel= " + pred.IsFAnovel() + " Recurrent : "
								+ pred.isRecurrent() + "  The Noelty Type: " + pred.getNoveltyType() + " Size: "
								+ pred.getClassSize();
						// s+="\n Density: "+
						// pred.getDenNovFlag()+" Gravity: "+
						// pred.getGravNovFlag()+"\n";
						if (pred.getNoveltyType().contains("Tp")) {
							strTP = Integer.toString(pred.getClassSize());
							for (int md = 0; md < CEModel.getNumberOfClasses(); md++) {

								s += " Max Distance= " + CEModel.getMaxDistance(md);
							}

						}
						double av = Double.parseDouble(pred.getClassTrueLabel()[1]) / pred.getClassSize();
						avPurity += av;
						pCount++;

						if (pred.IsFAnovel() || pred.isRecurrent()) {
							if (pred.IsFAnovel()) {
								strFA = Integer.toString(pred.getClassSize());
								fa_counter++;
								fa_ptsCounter += pred.getClassSize();
							}
							if (pred.isRecurrent()) {
								strTR = Integer.toString(pred.getClassSize());
								rec_counter++;
								rec_ptscounter += pred.getClassSize();

							}
							s += "\n Novel Class Size: ";
							s += pred.getClassSize();
							s += " Class majority label: ";
							s += pred.getClassTrueLabel()[0];
							s += " Purity: " + pred.getClassTrueLabel()[1] + " Max Distance= "
									+ CEModel.getMaxDistance(CEModel.getNumberOfClasses() - 1);
						} else if (pred.getNoveltyType().contains("Fn")) {
							strFN = Integer.toString(pred.getClassSize());
							s += "Novel class classified as existing ";

						}
						if (pred.isUnknown()) {
							strUN = Integer.toString(pred.getClassSize());
							unknown += pred.getClassSize();
							unknownC++;
							if (CEModel.getUpateFlag()) {

								s += CEModel.updateModelConceptEvolution_Unknown(CEModel.getDeclaredData(), pred);
								updateStr += i + ",";
							}

						}
						s += "\n";
						if (pred.getNoveltyType().contains("Fn")) {
							confusionMatrix[CEModel.getOldLabels().length][ID(pred.getPredictedLabel(),
									CEModel.getOldLabels())] += pred.getClassSize();
						}
						if (pred.getNoveltyType().contains("Tn")) {
							strTN = Integer.toString(pred.getClassSize());
							confusionMatrix[ID(pred.getClassTrueLabel()[0], CEModel.getOldLabels())][ID(
									pred.getPredictedLabel(), CEModel.getOldLabels())] += pred.getClassSize();
						}
						if (pred.getNoveltyType().contains("Fp")) {
							strFP = Integer.toString(pred.getClassSize());
							confusionMatrix[ID(pred.getClassTrueLabel()[0],
									CEModel.getOldLabels())][CEModel.getOldLabels().length + 1] += pred.getClassSize();
						}

						NovelStats.add(pred);
						outBuff.append(pred.getDetails());
						outBuff.append(s);
						sdet = segSize + "," + i + "," + p + "," + strTP + "," + strTN + "," + strFP + "," + strFN + ","
								+ strFA + "," + strTR + "," + strUN;
						detARR.add(sdet);

						sdet = "";
						strTP = "";
						strTN = "";
						strFP = "";
						strFN = "";
						strFA = "";
						strTR = "";
						strUN = "";
						if (validFlag) {

							s += CEModel.updateModelConceptEvolution(pred);
							updateStr += i + ",";

							if (s.contains("No-Update") || s.contains("No-Change")) {
								updateStr += "0 \n";
								activeCounter.add(-1);
							} else if (s.contains("Update-Model")) {
								updateStr += "1 \n";
								updatedCounter.add(i);
							} else
								updateStr += "-1 \n";

						}

					}

					counter_w++;
					fileOutStr += counter_w + ",";
					fileOutStr += NP;
				}
			}

			// This is updated.
			// To use only labels that are existing in the learning
			// model.
			// if new labels detected they are not added to the
			// model
			//
			String[] old_labels = new String[AssignmentArr.length];

			double[] pointsCounter = new double[AssignmentArr.length];
			int count = 0;
			// Find the majority label for test the purity of
			// testing instances
			for (int p = 0; p < old_labels.length; p++) {
				if (AssignmentArr[p].numInstances() == 0)
					break;
				String[] clusterLabel = majorityLabel(AssignmentArr[p], classValues);
				for (int h = 0; h < CEModel.getLabels().length; h++) {
					if (CEModel.getLabels()[h].contains(clusterLabel[0])) {
						old_labels[p] = clusterLabel[0];
						count++;
						pointsCounter[p] = Double.parseDouble(clusterLabel[1]);
					}

				}
			}

			String[] labels = new String[count];
			int k = 0;
			for (int m = 0; m < old_labels.length; m++) {
				if (old_labels[m] != null) {
					labels[k] = old_labels[m];
					k++;
				}
			}

		}

		StringBuffer res = printStats(NovelStats, unknown, confusionMatrix, CEModel, updatedCounter, activeCounter);
		FileWriter fstream = new FileWriter(m_Instances.relationName() + ".csv");
		BufferedWriter out = new BufferedWriter(fstream);
		for (int y = 0; y < detARR.size(); y++)
			out.write(detARR.get(y) + "\n");
		System.out.println(fileOutStr);
		avPurity /= pCount;

		System.out.println("Average Purity in clusters = " + avPurity * 100 + " %  in " + pCount + "  Clusters");
		System.out.println("Results summary and accuracy \n ------------");

		System.out.println(res);

		out.close();
		return CEModel;
	}

	private static void updateResults(String NP, NovelPredection pred) {
		if (NP.contains("Case 0.0"))
			s += "**buffer size ";
		else
			s += "**Release ";
		s += " True label= " + pred.getClassTrueLabel()[0] + " Predicted label= " + pred.getPredictedLabel()
				+ " FA novel= " + pred.IsFAnovel() + " Recurrent Novel= " + pred.isRecurrent() + "  The Noelty Type: "
				+ pred.getNoveltyType() + " Size = " + pred.getClassSize();
		// s += "\n Density: " +
		// pred.getDenNovFlag()
		// + " Gravity: " + pred.getGravNovFlag()
		// + "\n";
		if (pred.getNoveltyType().contains("Tp")) {
			strTP = Integer.toString(pred.getClassSize());
			for (int md = 0; md < CEModel.getNumberOfClasses(); md++) {

				s += " Max Distance= " + CEModel.getMaxDistance(md) + " ";
			}

		}
		double av = Double.parseDouble(pred.getClassTrueLabel()[1]) / pred.getClassSize();
		avPurity += av;
		pCount++;
		if (pred.IsFAnovel() || pred.isRecurrent()) {
			if (pred.IsFAnovel()) {
				strFA = Integer.toString(pred.getClassSize());
				fa_counter++;
				fa_ptsCounter += pred.getClassSize();
			}
			if (pred.isRecurrent()) {
				strTR = Integer.toString(pred.getClassSize());
				rec_counter++;
				rec_ptscounter += pred.getClassSize();
			}
			s += "\n Novel Class Size: ";
			s += pred.getClassSize();
			s += " Class majority label: ";
			s += pred.getClassTrueLabel()[0];
			s += " Purity: " + pred.getClassTrueLabel()[1] + " Max Distance= "
					+ CEModel.getMaxDistance(CEModel.getNumberOfClasses() - 1);
		} else if (pred.getNoveltyType().contains("Fn")) {
			strFN = Integer.toString(pred.getClassSize());
			s += "Novel class classified as existing";

		}
		if (pred.isUnknown()) {
			strUN = Integer.toString(pred.getClassSize());
			unknown += pred.getClassSize();
			unknownC++;
			if (CEModel.getUpateFlag()) {

				s += CEModel.updateModelConceptEvolution_Unknown(CEModel.getDeclaredData(), pred);
				updateStr += i + ",";
			}

		}
		s += "\n";
		if (pred.getNoveltyType().contains("Fn")) {
			confusionMatrix[CEModel.getOldLabels().length][ID(pred.getPredictedLabel(), CEModel.getOldLabels())] += pred
					.getClassSize();
		} else if (pred.getNoveltyType().contains("Tn")) {
			strTN = Integer.toString(pred.getClassSize());
			confusionMatrix[ID(pred.getClassTrueLabel()[0], CEModel.getOldLabels())][ID(pred.getPredictedLabel(),
					CEModel.getOldLabels())] += pred.getClassSize();
		} else if (pred.getNoveltyType().contains("Fp")) {
			strFP = Integer.toString(pred.getClassSize());
			confusionMatrix[ID(pred.getClassTrueLabel()[0], CEModel.getOldLabels())][CEModel.getOldLabels().length
					+ 1] += pred.getClassSize();
		} else if (pred.isUnknown()) {
			int sID = ID(pred.getClassTrueLabel()[0], CEModel.getOldLabels());
			if (sID != -1)
				confusionMatrix[sID][CEModel.getOldLabels().length] += pred.getClassSize();
		} else if (pred.getNoveltyType().contains("Tp")) {
			strTP = Integer.toString(pred.getClassSize());
			confusionMatrix[CEModel.getOldLabels().length][CEModel.getOldLabels().length + 1] += pred.getClassSize();
		}

		NovelStats.add(pred);
		sdet = segSize + "," + i + "," + p + "," + strTP + "," + strTN + "," + strFP + "," + strFN + "," + strFA + ","
				+ strTR + "," + strUN;
		detARR.add(sdet);

		sdet = "";
		strTP = "";
		strTN = "";
		strFP = "";
		strFN = "";
		strFA = "";
		strTR = "";
		strUN = "";
	}

	public static Instances removeClass(Instances inst) {
		Remove af = new Remove();
		Instances retI = null;

		try {
			if (inst.classIndex() < 0) {
				retI = inst;
			} else {
				af.setAttributeIndices("" + (inst.classIndex() + 1));
				af.setInvertSelection(false);
				af.setInputFormat(inst);
				retI = Filter.useFilter(inst, af);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return retI;
	}

	private static String[] majorityLabel(Instances ins, String[] classValues) {
		// TODO Auto-generated method
		double[] counter = new double[ins.numClasses()];
		String[] clusterLabels = new String[ins.numClasses()];
		int f = 0;
		for (Enumeration<?> c = ins.classAttribute().enumerateValues(); c.hasMoreElements();) {

			clusterLabels[f] = (String) c.nextElement();
			f++;
		}
		ins.setClassIndex(ins.numAttributes() - 1);
		for (int i = 0; i < ins.numInstances(); i++) {
			double s = ins.instance(i).classValue();
			if (s != -1)
				counter[(int) s]++;

		}
		double max = 0;
		int s = -1;
		int n = 0;
		for (int i = 0; i < counter.length; i++) {
			n += counter[i];
			while (counter[i] > max) {
				s = i;
				max = counter[i];
			}
		}
		double perc = max / n;
		String[] clusterLabel = new String[2];
		clusterLabel[0] = clusterLabels[s];
		clusterLabel[1] = Double.toString(perc);

		return clusterLabel;

	}

	private static StringBuffer printStats(ArrayList<NovelPredection> Arr, int unknown, double[][] confusionMatrix,
			BLM CEModel, ArrayList<Integer> updatedCounter, ArrayList<Integer> activeCounter) {
		double nInstances = 0;
		double Tp = 0, Tn = 0, Fp = 0, Fn = 0;
		double pt_Tp = 0, pt_Tn = 0, pt_Fp = 0, pt_Fn = 0;
		StringBuffer results = new StringBuffer();
		int fa_counter = 0;
		int fa_ptsCounter = 0;
		int rec_counter = 0;
		int rec_ptscounter = 0;

		for (int nArr = 0; nArr < Arr.size(); nArr++) {
			NovelPredection NP = Arr.get(nArr);
			nInstances += NP.getClassSize();
			// System.out.println(nInstances);
			if (NP.getNoveltyType().contains("Tp")) {
				Tp += NP.getClassSize();
				pt_Tp += NP.getNoofNovelInsatces();
				pt_Fp += NP.getClassSize() - NP.getNoofNovelInsatces();
			} else if (NP.getNoveltyType().contains("Tn")) {
				Tn += NP.getClassSize();
				pt_Tn += NP.getClassSize() - NP.getNoofNovelInsatces();
				pt_Fn += NP.getNoofNovelInsatces();
			} else if (NP.getNoveltyType().contains("Fp")) {
				Fp += NP.getClassSize();
				pt_Fp += NP.getClassSize() - NP.getNoofNovelInsatces();
				pt_Tp += NP.getNoofNovelInsatces();
			} else if (NP.getNoveltyType().contains("Fn")) {
				Fn += NP.getClassSize();
				pt_Fn += NP.getNoofNovelInsatces();
				pt_Tn += NP.getClassSize() - NP.getNoofNovelInsatces();
			}

		}

		results.append(" ============CLUSTER BASED MEASURES==========\n");

		results.append("Total Number of Instances: " + nInstances + " Unknown Instances= " + unknown + ", "
				+ (unknown / nInstances) + " %\n");

		results.append("TP  (instances) = " + Tp + "\n");
		results.append("TN  (instances)= " + Tn + "\n");
		results.append("FP (instances) = " + Fp + "\n");
		results.append("FN (instances)= " + Fn + "\n");
		results.append("Accuracy (Excl. unknown)      = " + (Tp + Tn) * 100 / (nInstances - unknown) + " %\n");
		results.append("Accuracy  (all)      = " + (Tp + Tn) * 100 / (nInstances) + "%\n");
		results.append("Recall         = " + Tp * 100 / (Tp + Fn) + " %\n");
		results.append("Specifity	   = " + Tn * 100 / (Tn + Fp) + " %\n");
		results.append("Perception     = " + Tp * 100 / (Tp + Fp) + " %\n");
		results.append("Fall out rate   = " + Fp * 100 / (Fp + Tn) + " %\n");
		results.append("False discovery rate   = " + Fp * 100 / (Fp + Tp) + " %\n");
		results.append("F1 Score    = " + (2 * Tp) * 100 / ((2 * Tp) + Fp + Fn) + " %\n");
		results.append(" \n Error (incl. Fp, Fn and unknown) = " + ((unknown + Fp + Fn) * 100) / nInstances + " %\n");

		results.append("\n============ Point BASED MEASURES=========\n");
		results.append("TP  = " + pt_Tp + "\n");
		results.append("TN  = " + pt_Tn + "\n");
		results.append("FP = " + pt_Fp + "\n");
		results.append("FN = " + pt_Fn + "\n");
		results.append("Accuracy (Excl. Unknown)   = " + (pt_Tp + pt_Tn) * 100 / (nInstances - unknown) + " %\n");
		results.append("Accuracy  (all)      = " + (pt_Tp + pt_Tn) * 100 / (nInstances) + "%\n");
		results.append("Recall         = " + pt_Tp * 100 / (pt_Tp + pt_Fn) + " %\n");
		results.append("Specifity      = " + pt_Tn * 100 / (pt_Tn + pt_Fp) + " %\n");
		results.append("Perception     = " + pt_Tp * 100 / (pt_Tp + pt_Fp) + " %\n");
		results.append("Fall out rate   = " + pt_Fp * 100 / (pt_Fp + pt_Tn) + " %\n");
		results.append("False discovery rate   = " + pt_Fp * 100 / (pt_Fp + pt_Tp) + " %\n");
		results.append("F1 Score    = " + (2 * pt_Tp) * 100 / ((2 * pt_Tp) + pt_Fp + pt_Fn) + " %\n");

		results.append(
				" \n Error (incl. Fp, Fn and unknown) = " + ((unknown + pt_Fp + pt_Fn) * 100) / nInstances + " %\n");

		int accuracy = 0;
		results.append(" ======Confusion Matrix==============\n");
		results.append("Total Number of predicted Instances=  " + (Tn + Fp) + "\n");
		String[] m_Labels = CEModel.getOldLabels();
		for (int index = 0; index < m_Labels.length; index++)
			results.append("		" + m_Labels[index]);
		results.append("		Unknown");
		results.append("		Novel");
		results.append("\n");
		for (int index = 0; index < m_Labels.length + 1; index++) {
			if (index != m_Labels.length)
				results.append(m_Labels[index]);
			else
				results.append("Novel");
			for (int j = 0; j < m_Labels.length + 2; j++) {
				results.append("		" + confusionMatrix[index][j]);
				if (index == j)
					accuracy += confusionMatrix[index][j];
			}
			results.append("\n");
		}
		results.append("Predection accuracy= " + (accuracy * 100) / (Tn + Fp) + "%\n \n-------------\n");
		// ////////////////////////////////////////////////
		// Cases
		int CaseCounter = 0;
		int[][] cases = CEModel.getCasesCounter();
		for (int j = 0; j < 6; j++) {
			CaseCounter += cases[0][j];
			results.append("Occurrences of case 0." + j + " = " + cases[0][j] + "\n");
		}
		// outBuff.append("---------------------------");
		for (int j = 1; j < 4; j++) {
			CaseCounter += cases[1][j];
			results.append("Occurances of case 1." + j + " = " + cases[1][j] + "\n");
		}
		// outBuff.append("---------------------------");
		for (int j = 1; j < 4; j++) {
			CaseCounter += cases[2][j];
			results.append("Occurrences of case 2." + j + " = " + cases[2][j] + "\n");
		}
		// outBuff.append("---------------------------");
		for (int j = 0; j < 5; j++) {
			CaseCounter += cases[3][j];
			results.append("Occurrences of case 3." + j + " = " + cases[3][j] + "\n");
		}
		// outBuff.append("---------------------------");
		for (int j = 1; j < 5; j++) {
			CaseCounter += cases[4][j];
			results.append("Occurrences of case 4." + j + " = " + cases[4][j] + "\n");
		}
		results.append("TOTAL cases: " + CaseCounter + "\n");
		results.append(
				"Icremental and Active Total: " + CEModel.getIncCounter() + " , " + updatedCounter.size() + "\n");
		results.append("Active only total: " + CEModel.actLearningRate() + " / Cases Fp= "
				+ CEModel.actLearningCases()[0] + " Recurrent= " + CEModel.actLearningCases()[1] + " Novel= "

				+ CEModel.actLearningCases()[2] + "\n");

		results.append("total fa Novel : " + fa_counter + " pts=  " + fa_ptsCounter + "  Recurrent  " + rec_counter
				+ " Total recurrent points   " + rec_ptscounter + "\n");
		results.append(
				"Unknown cases, case inpurity: " + CEModel.unKnownStats()[0] + " / Cases existing / update model = "
						+ CEModel.unKnownStats()[1] + " Recurrent= " + CEModel.unKnownStats()[2] + " Novel= "

						+ CEModel.unKnownStats()[3] + "\n");

		return results;
	}

	// return the ID of the predicted label, if unknown it returns
	// labels.length+1

	private static int ID(String str, String[] labels) {

		int i = 0;
		for (i = 0; i < labels.length; i++) {
			if (labels[i].trim().toLowerCase().contains(str.trim().toLowerCase()))
				return i;
		}
		return -1;
	}

}
