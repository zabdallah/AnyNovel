package conceptEvolution;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import StreamAR.BLM;
import StreamAR.ClassWSubClusters;
import StreamAR.Model;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

public class ExpLauncher {
	// To run training/ building the model followed by AnyNovel using the built
	// model.
	static long startTime;
	static long duration;
	protected static Model baseModel = null;

	public static void main(String[] args) throws Exception {
		// ------- Collecting arguments and build the model
		BLM CWSCModel = null;
		boolean clusOption = false;
		String str = "-clust";
		for (String item : args) {
			if (str.equalsIgnoreCase(item)) {
				clusOption = true;
				break; // No need to look further.
			}
		}
		int ind = 0;
		str = "-blm";
  		for (int i = 0; i < args.length; i++) {
			if (str.equalsIgnoreCase(args[i])) {
				ind = i;
				break;

			}
		}
		if (args.length > 2) {
			if (args[ind].toLowerCase().trim().contains("-blm")) {
				if (args[ind + 1].toLowerCase().trim().contains("train")) {
					System.out.println("Build the model");
					if (clusOption)
						CWSCModel = TrainingLauncher.buildModel(args[ind + 2], args[ind + 4], args[ind + 6]);
					else
						CWSCModel = TrainingLauncher.buildModel(args[ind + 2], "", "");

				} else if (args[1].toLowerCase().trim().contains("load")) {
					System.out.println("Load Model .....");
					CWSCModel = readModel(args[2]);
				}
			}
		} else {
			System.out.println("arguments required: -BLM train/load dataset_name");
		}

		// -------- Run AnyNovel on the provided test file
		String testName = findArg(args, "-test");
		String validName = findArg(args, "-valid");
		startTime = System.currentTimeMillis();
		AnyNovelLauncher.run(CWSCModel, testName, validName);
		duration = System.currentTimeMillis() - startTime;
		System.out.println("Duration of AnyNovel: " + Utils.doubleToString(duration / 1000.0, 10));
	}

	private static String findArg(String[] args, String str) {
		for (int i = 0; i < args.length; i++) {
			if (str.equalsIgnoreCase(args[i])) {
				return args[i + 1];

			}
		}
		return "";
	}

	static BLM readModel(String s) throws Exception {
		baseModel = null;
		BLM CWSCModel = null;
		String filePath = new File("").getAbsolutePath();

		File fc = new File(filePath.concat("/Models/" + s));

		if (fc.getName().contains(".dat")) {
			BufferedReader in = new BufferedReader(new FileReader(fc.getPath()));
			CWSCModel = readFromDatFile(in);
		} else {
			FileInputStream fileIn = new FileInputStream(fc.getPath());
			ObjectInputStream in = new ObjectInputStream(fileIn);
			baseModel = (Model) in.readObject();
			in.close();
			fileIn.close();
		}

		StringBuffer outBuff = new StringBuffer();
		String logName = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
		logName += "File Model Builder Deserialization";
		if (fc.getName().contains("_CWSCM") || fc.getName().contains("dat")) {
			outBuff.append("\n=== subclusters Base Model from file ===\n");
			outBuff.append(fc.getName() + "\n");

			outBuff.append(CWSCModel.modelStatistices());

		} else {
			outBuff.append("\n=== Basic Base Model from file ===\n");

			outBuff.append(baseModel.modelStatistices());
		}
		System.out.println(outBuff);
		return CWSCModel;

	}

	private static BLM readFromDatFile(BufferedReader in) throws IOException {

		String fileStr = "";

		int noOfClasses = Integer.parseInt(in.readLine().trim());
		// fileStr += cWSCModel.getNumberOfClasses() + "\n";
		ArrayList<ClassWSubClusters> Arr = new ArrayList<ClassWSubClusters>();
		for (int i = 0; i < noOfClasses; i++) {

			String label = in.readLine().trim();
			int NoOfSubClusters = Integer.parseInt(in.readLine().trim());
			// fileStr += Arr.get(i).getNoSubClusters() + "\n";
			double[][] gravForce = new double[NoOfSubClusters][NoOfSubClusters];
			for (int j = 0; j < NoOfSubClusters; j++) {
				for (int k = 0; k < NoOfSubClusters; k++) {
					gravForce[j][k] = Double.parseDouble(in.readLine().trim());
				}
			}

			double globalAvDistance = Double.parseDouble(in.readLine().trim());
			// fileStr+=
			// Arr.get(i).getGlobalAvDistance()+"\n";
			double globalBoundry = Double.parseDouble(in.readLine().trim());
			// fileStr += Arr.get(i).getClassBoundry() + "\n";

			int classID = Integer.parseInt(in.readLine().trim());
			// fileStr+=Arr.get(i).getClassID()+"\n";

			double farthest_dis = Double.parseDouble(in.readLine().trim());
			// fileStr+= Arr.get(i).getFarthestdis()+"\n";
			double maxVDL = Double.parseDouble(in.readLine().trim());
			// fileStr+= Arr.get(i).getMaxVDL()+"\n";
			int totalsize = Integer.parseInt(in.readLine().trim());
			// fileStr+= Arr.get(i).getTotalsize()+"\n";
			double minGravF = Double.parseDouble(in.readLine().trim());
			// fileStr+= Arr.get(i).getMinGravF()+"\n";

			double[] m_classBoundry = new double[NoOfSubClusters];
			double[] m_averageDistance = new double[NoOfSubClusters];
			double[] VDL = new double[NoOfSubClusters];
			int[] size = new int[NoOfSubClusters];

			for (int j = 0; j < NoOfSubClusters; j++) {
				m_classBoundry[j] = Double.parseDouble(in.readLine().trim());
				m_averageDistance[j] = Double.parseDouble(in.readLine().trim());
				VDL[j] = Double.parseDouble(in.readLine().trim());
				size[j] = Integer.parseInt(in.readLine().trim());
				// fileStr += Arr.get(i).getBoundries()[j] + "\n";
				// fileStr += Arr.get(i).getAvDistances()[j] + "\n";
				// fileStr += Arr.get(i).getVDL()[j] + "\n";
				// fileStr += Arr.get(i).getSizes()[j] + "\n";
			}
			int nAtt = Integer.parseInt(in.readLine().trim());
			// int nAtt = Arr.get(i).getCentres()[0]
			// .numAttributes();
			// fileStr += nAtt + "\n";
			// Instance classSD
			fileStr = in.readLine().trim();
			String[] classSDStr = fileStr.split(",");
			Instance classSD = convertStringArrtoInst(classSDStr);

			fileStr = in.readLine().trim();
			String[] classcntrStr = fileStr.split(",");
			Instance classCentre = convertStringArrtoInst(classcntrStr);

			Instance[] SD = new Instance[NoOfSubClusters];
			Instance[] m_centres = new Instance[NoOfSubClusters];
			for (int j = 0; j < NoOfSubClusters; j++) {
				fileStr = in.readLine().trim();
				String[] strArr = fileStr.split(",");
				Instance inst = convertStringArrtoInst(strArr);
				m_centres[j] = (Instance) inst.copy();
			}
			for (int j = 0; j < NoOfSubClusters; j++) {
				fileStr = in.readLine().trim();
				String[] strArr = fileStr.split(",");
				Instance inst = convertStringArrtoInst(strArr);
				SD[j] = (Instance) inst.copy();
			}

			ClassWSubClusters fileObj = new ClassWSubClusters(m_centres, size, SD, m_classBoundry, m_averageDistance,
					NoOfSubClusters, label, classID, classSD, farthest_dis, globalAvDistance, globalBoundry, gravForce,
					maxVDL, minGravF, size, totalsize, VDL, classCentre);
			Arr.add(fileObj);
		}

		String[] Labels = new String[noOfClasses];
		for (int i = 0; i < noOfClasses; i++)
			Labels[i] = in.readLine().trim();
		// fileStr += cWSCModel.getLabels()[i] + "\n";
		BLM fileModel = new BLM(Arr, noOfClasses, Labels);
		return fileModel;

	}

	private static Instance convertStringArrtoInst(String[] strArr) {
		double[] InstanceArr = convertStringArrtoDouble(strArr);
		ArrayList<Attribute> atts = new ArrayList<Attribute>(InstanceArr.length);
		for (int i = 0; i < InstanceArr.length; i++) {
			atts.add(new Attribute("att" + i));
		}

		// Instances dataRaw = new Ins
		//// new Instances("TestInstances", atts, 0);

		int j;
		Instance ins = new Instance(InstanceArr.length);
		// ins.setDataset(dataRaw);
		for (j = 0; j < InstanceArr.length; j++) {
			ins.setValue(j, InstanceArr[j]);
		}

		// dataRaw.add((Instance) ins.copy());

		return ins;
	}

	private static double[] convertStringArrtoDouble(String[] strArr) {
		double[] arr = new double[strArr.length];
		for (int i = 0; i < strArr.length; i++)
			arr[i] = Double.parseDouble(strArr[i]);

		return arr;
	}

}
