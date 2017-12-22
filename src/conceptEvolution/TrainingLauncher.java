package conceptEvolution;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.Scanner;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import StreamAR.ClassWSubClusters;
import StreamAR.IntCWSCClass;
import StreamAR.Model;
import StreamAR.BLM;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Attribute;
//import weka.core.DenseInstance;
import weka.core.FastVector;
//import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;

public class TrainingLauncher {
	// For either building a model from a training data or uploading an existing
	// model
	// To read the training data, then build the model offline. The model is
	// then saved in the output folder
	// Read CSV and arff file, class index is the last column
	// If the model is provided, we only set the model and then move to anyNovel
	// prediction phase
	protected boolean subClusterModelFlag = false;
	protected static Model baseModel = null;
	private static Instances m_Instances;
	boolean CWSCFlag = true;

	protected int old_distance = 0;

	public static void main(String[] args) throws Exception {
		boolean clusOption = false;
		String str = "-clust";
		for (String item : args) {
			if (str.equalsIgnoreCase(item)) {
				clusOption = true;
				break; // No need to look further.
			}
		}
		BLM CWSCModel = null;
		if (args.length > 2) {
			if (args[0].toLowerCase().trim().contains("-blm")) {
				if (args[1].toLowerCase().trim().contains("train")) {
					System.out.println("Build the model");
					if (clusOption)
						CWSCModel = buildModel(args[2], args[4], args[6]);
					else
					{
						System.out.println(args[2]);
						CWSCModel = buildModel(args[2], "", "");
					}
				} 
				else {
					System.out.println(
							"arguments required: -BLM train dataset_name, "
							+ "load model only available through ExpLauncher. "
							+ "Train Launcher is only for Building a model and storing it into models");
				}
			
		} else {
			System.out.println("arguments required: -BLM train dataset_name");
		}
		}
	}

	static BLM buildModel(String s, String clusterName, String Nclusters) throws Exception {

		BLM CWSCModel = null;
		File trainFile = null;
		String datasetname = "";
		// Define relative path
		String filePath = new File("").getAbsolutePath();
		// 1- option 1: if the exact file provided "Including extension
		if (s.trim().toLowerCase().contains("csv") || s.trim().toLowerCase().contains("arff")) {
			System.out.println(s);
			datasetname = s.split("\\.")[0];
			System.out.println("Dataset name: "+datasetname);

			filePath = filePath.concat("/Data/Train/" + s);
			trainFile = new File(filePath);
			
			System.out.println("Training File Path: "+trainFile);

			if (new File(trainFile.getAbsolutePath()).exists()) {
			} else {
				System.out.println("File is not exist");
				return null;
			}
			// if (new File(trainFile.getAbsolutePath() + ".csv").exists()) {
			// filePath = trainFile + ".csv";
			// } else if (new File(trainFile.getAbsolutePath() +
			// ".arff").exists()) {
			// filePath = trainFile + ".arff";
			// }
		} else {
			//Option 2: dataset name 
			datasetname = s;
			filePath = filePath.concat("/Data/Train/");
			File folder = new File(filePath);
			File[] listOfFiles = folder.listFiles();
			System.out.println("Files in the direcory");
			int index = 0;
			ArrayList<Integer> indexArr = new ArrayList<Integer>();

			for (int i = 0; i < listOfFiles.length; i++) {
				if (listOfFiles[i].isFile() & listOfFiles[i].getName().contains(datasetname)) {
					System.out.println(index + " : " + listOfFiles[i].getName());
					index++;
					indexArr.add(i);

				}
			}
			Scanner reader = new Scanner(System.in); // Reading from System.in
			System.out.println("Enter the number of training file: ");
			int n = reader.nextInt(); // Scans the next token of the input as an
										// int
			trainFile = new File(filePath + listOfFiles[indexArr.get(n)].getName());
			filePath = trainFile.getAbsolutePath();

		}
		System.out.println(filePath);
		DataSource source = new DataSource(filePath);
		m_Instances = source.getDataSet();
		// setting class attribute if the data format does not provide this
		// information
		// For example, the ARFF format saves the class attribute information as
		// well
		if (m_Instances.classIndex() == -1)
			m_Instances.setClassIndex(m_Instances.numAttributes() - 1);

		long trainTimeStart = 0;
		long trainTimeElapsed = 0;
		StringBuffer outBuff = new StringBuffer();

		outBuff.append("\n=== Clustering Base Model data ===\n");

		// set the training instances
		Instances m_trainInstaces = new Instances(m_Instances);
		m_trainInstaces.setClassIndex(m_trainInstaces.numAttributes() - 1);
		m_trainInstaces.setRelationName(datasetname);
		// Build Class with sub clusters Model from train instances
		CWSCModel = buildCWSCModel(m_trainInstaces, clusterName, Nclusters);
		System.gc();
		Runtime.getRuntime().gc();

		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
		try {
			outBuff.append(CWSCModel.modelStatistices());
		} catch (Exception e) {
			e.printStackTrace();
		}
		outBuff.append("\nTime taken to build CWSC model : " + Utils.doubleToString(trainTimeElapsed / 1000.0, 10)
				+ " seconds\n\n --------------------------------\n");
		System.out.println(outBuff);
		return CWSCModel;
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

	private static BLM buildCWSCModel(Instances m_trainInstaces, String clusterName, String nclusters)
			throws FileNotFoundException, IOException {

		// Main Function to build cluster with sub clusters model as baseline
		// model from the
		// training data
		BLM CWSCModel = null;
		ArrayList<ClassWSubClusters> CWSCArr = new ArrayList<ClassWSubClusters>();
		ArrayList<Instances> m_model = new ArrayList<Instances>();
		ArrayList<Instances> classSubClusters = new ArrayList<Instances>();
		Instances[] m_modelInsts;

		// initialising all
		Attribute classAttr = m_trainInstaces.classAttribute();
		m_modelInsts = new Instances[classAttr.numValues()];

		// create attrInfo FastVictor
		FastVector attrInfo = new FastVector();

		for (Enumeration<?> enumAttr = m_trainInstaces.enumerateAttributes(); enumAttr.hasMoreElements();) {
			attrInfo.addElement((Attribute) enumAttr.nextElement());
		}
		attrInfo.addElement(m_trainInstaces.classAttribute());

		// initiate new instances
		for (int i = 0; i < m_modelInsts.length; i++) {
			m_modelInsts[i] = new Instances(classAttr.value(i), attrInfo, 0);
			m_modelInsts[i].setClassIndex(m_trainInstaces.classIndex());
		}
		// m_modelInsts Separate instances into groups. Each group represents a
		// label (from the ground truth, labelled data)
		for (Enumeration<?> e = m_trainInstaces.enumerateInstances(); e.hasMoreElements();) {
			Instance current = (Instance) e.nextElement();
			String name = current.stringValue(current.numAttributes() - 1);
			for (int i = 0; i < m_modelInsts.length; i++) {
				if (classAttr.value(i) == name) {
					m_modelInsts[i].add(current);
					break;
				}
			}
		}
		// create a clean version of m_modelInsts that ignores classes that have
		// no members >> m_model
		for (int i = 0; i < m_modelInsts.length; i++) {
			if (m_modelInsts[i].numInstances() != 0) {
				m_model.add(m_modelInsts[i]);

			}
		}
		// For each class, build clusters insides
		String[] labels = new String[m_model.size()];
		for (int j = 0; j < m_model.size(); j++) {
			// Cluster inside class
			classSubClusters = initSubClusters(m_model.get(j), clusterName, nclusters);
			Instance c = (Instance) classSubClusters.get(0).firstInstance().copy();
			labels[j] = c.stringValue(c.numAttributes() - 1);
			IntCWSCClass intClass = new IntCWSCClass(classSubClusters, labels[j], j);
			ClassWSubClusters obj = new ClassWSubClusters(intClass.m_centres, intClass.m_size, intClass.m_SD,
					intClass.m_classBoundry, intClass.m_averageDistance, intClass.NoOfSubClusters, intClass.label,
					intClass.m_classID, intClass.m_classSD, intClass.m_farthest_dis, intClass.m_globalAvDistance,
					intClass.m_globalBoundry, intClass.m_gravForce, intClass.m_maxVDL, intClass.m_minGravF,
					intClass.m_size, intClass.m_totalsize, intClass.m_VDL, intClass.m_classCentre);
			CWSCArr.add(obj);
		}
		String fileName ="";
		System.out.println(fileName);

		for (int n = 0; n < labels.length; n++)
			fileName += "_" + labels[n];
		System.out.println(fileName);
		CWSCModel = new BLM(CWSCArr, CWSCArr.size(), labels);
		writeModelToFile(CWSCModel, fileName);
		return CWSCModel;

	}

	private static void writeModelToFile(BLM cWSCModel, String ext) {
		// Write the model in a specific format that can be read/decoded later
		// with readModel Function
		try {
			// Create file
			String path = new File("").getAbsolutePath();
			Scanner reader = new Scanner(System.in); // Reading from System.in
			System.out.println("Wrirting model to Disk (/Models/), Enter the name of your dataset: ");
			String  datasetName = reader.next(); // Scans the next token of the input as an
			datasetName+= ext; 
			FileWriter fstream = new FileWriter(path.concat("/Models/" + datasetName + ".dat"));
			BufferedWriter out = new BufferedWriter(fstream);
			System.out.println(datasetName+".dat created in /Models/");

			String fileStr = "";
			fileStr += cWSCModel.getNumberOfClasses() + "\n";
			ArrayList<ClassWSubClusters> Arr = cWSCModel.getClassWithSubClustersArray();
			for (int i = 0; i < Arr.size(); i++) {

				fileStr += Arr.get(i).getLable() + "\n";
				fileStr += Arr.get(i).getNoSubClusters() + "\n";
				for (int j = 0; j < Arr.get(i).getNoSubClusters(); j++) {
					for (int k = 0; k < Arr.get(i).getNoSubClusters(); k++) {
						fileStr += Arr.get(i).getGravForce()[j][k] + "\n";
					}
				}

				fileStr += Arr.get(i).getGlobalAvDistance() + "\n";
				fileStr += Arr.get(i).getClassBoundry() + "\n";
				fileStr += Arr.get(i).getClassID() + "\n";

				fileStr += Arr.get(i).getFarthestdis() + "\n";
				fileStr += Arr.get(i).getMaxVDL() + "\n";
				fileStr += Arr.get(i).getTotalsize() + "\n";
				fileStr += Arr.get(i).getMinGravF() + "\n";

				for (int j = 0; j < Arr.get(i).getNoSubClusters(); j++) {
					fileStr += Arr.get(i).getBoundries()[j] + "\n";
					fileStr += Arr.get(i).getAvDistances()[j] + "\n";
					fileStr += Arr.get(i).getVDL()[j] + "\n";
					fileStr += Arr.get(i).getSizes()[j] + "\n";
				}

				int nAtt = Arr.get(i).getCentres()[0].numAttributes();
				fileStr += nAtt + "\n";
				for (int k = 0; k < nAtt; k++) {
					fileStr += Arr.get(i).getClassSD().value(k);
					if (k < nAtt - 1)
						fileStr += ",";
				}
				fileStr += "\n";
				// classCentre
				for (int k = 0; k < nAtt; k++) {
					fileStr += Arr.get(i).getClassCentre().value(k);
					if (k < nAtt - 1)
						fileStr += ",";
				}
				fileStr += "\n";

				// m_centres
				// SD
				for (int j = 0; j < Arr.get(i).getNoSubClusters(); j++) {
					for (int k = 0; k < nAtt; k++) {
						fileStr += Arr.get(i).getCentres()[j].value(k);
						if (k < nAtt - 1)
							fileStr += ",";
					}

					fileStr += "\n";
				}
				for (int j = 0; j < Arr.get(i).getNoSubClusters(); j++) {
					for (int k = 0; k < nAtt; k++) {
						fileStr += Arr.get(i).getSD()[j].value(k);
						if (k < nAtt - 1)
							fileStr += ",";
					}

					fileStr += "\n";
				}
			}

			for (int i = 0; i < cWSCModel.getNumberOfClasses(); i++)
				fileStr += cWSCModel.getLabels()[i] + "\n";

			out.write(fileStr);

			// Close the output stream
			out.close();
		} catch (Exception e) {// Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}

	}

	private static ArrayList<Instances> initSubClusters(Instances instances, String clusterName, String nclusters) {
		StringBuffer outBuff = new StringBuffer();
		Instance current = instances.firstInstance();
		outBuff.append(
				current.stringValue(current.numAttributes() - 1) + " " + instances.numInstances() + " instances");
		Instances iniInst = new Instances(instances);

		// outBuff.append("\n=== Clustering Model data for subclusters===\n\n");

		// Clustering
		// Set here the number of clusters in each cluster to 3 using EM method,
		// this can be changed later
		String[] attribNames = new String[iniInst.numAttributes()];
		iniInst.setClassIndex(attribNames.length - 1);
		String[] options = new String[2];
		options[0] = "-N"; // number of clusters
		if (nclusters == "")
			nclusters = "3";
		options[1] = nclusters;
		Clusterer clusterer = null;
		if (clusterName == "" || clusterName.toLowerCase().trim().contains("em")) {
			clusterer = new EM();
			try {
				((EM) clusterer).setOptions(options); // set the options
				clusterer.buildClusterer(removeClass(iniInst));
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		} else if (clusterName.toLowerCase().trim().contains("kmeans")) {
			clusterer = new SimpleKMeans();
			try {
				((SimpleKMeans) clusterer).setNumClusters(Integer.valueOf(nclusters));
				clusterer.buildClusterer(removeClass(iniInst));
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

		ClusterEvaluation eval2 = new ClusterEvaluation();

		eval2.setClusterer(clusterer);
		// m_Log.statusMessage("Clustering training data...");
		try {
			eval2.evaluateClusterer(iniInst, "", false);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		// outBuff.append(eval2.clusterResultsToString());
		// m_History.updateResult(logName);
		double[] Assignments = eval2.getClusterAssignments();

		// Return the biggest
		ArrayList<Instances> subClusters = getSubClusters(iniInst, Assignments, eval2.getNumClusters());
		System.out.println(outBuff);
		return subClusters;

	}

	private static ArrayList<Instances> getSubClusters(Instances iniInst, double[] assignments, int k) {
		Instances[] subClusterArr = new Instances[k];

		ArrayList<Instances> subClusters = new ArrayList<Instances>();

		for (int i = 0; i < k; i++) {
			subClusterArr[i] = new Instances(iniInst, iniInst.numInstances());
		}
		for (int i = 0; i < iniInst.numInstances(); i++)

			subClusterArr[(int) assignments[i]].add(iniInst.instance(i));

		for (int i = 0; i < k; i++) {
			subClusterArr[i].compactify();
			subClusters.add(subClusterArr[i]);
		}
		return subClusters;

	}
}
