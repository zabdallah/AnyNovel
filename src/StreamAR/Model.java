package StreamAR;

import java.io.Reader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;
import javax.swing.text.Utilities;
import java.lang.Object;
//import com.sun.corba.se.impl.util.Utility;

import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.estimators.NNConditionalEstimator;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Model implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5962230084262446166L;
	// No of classes in the model
	protected int m_noClasses;
	// protected double[] m_classDensity;
	protected Instance[] m_centres;
	protected int[] m_size;
	protected Instance[] m_SD;
	// protected double[][]centresArr;
	protected double m_sysCapacity;
	protected String[] labels;
	protected double m_gravF;
	protected double[] m_classBoundry;
	protected String predection[];
	protected int predStatus[];
	double densThreshold;
	int parentClassID = -1;
	protected double[] m_averageDisatnce;

	// List of instances
	// protected ArrayList<Instances> m_model= new ArrayList<>();

	public Model(ArrayList<Instances> m_model, int n, String[] clsLabels) {
		// constructor for clustered Instances , clusters are stored in Array
		// List
		m_noClasses = n;
		m_centres = getCentres(m_model);
		m_classBoundry = getClassBoundry(m_model);
		m_averageDisatnce = getAvDistance(m_model);
		m_size = new int[m_noClasses];
		m_size = getNoInstaces(m_model);
		densThreshold = 1000;
		// densThreshold= m_classBoundry[0];
		// for( int i=1; i< m_noClasses;i++){
		// if(densThreshold> m_classBoundry[i])
		// densThreshold= m_classBoundry[i];
		// }

		// m_classDensity= new double[m_noClasses];
		// m_classDensity= this.getDensity(m_model);
		m_SD = getSD(m_model);
		m_sysCapacity = 0;
		for (int j = 0; j < this.m_noClasses; j++) {
			m_sysCapacity += m_model.get(j).numInstances();
		}
		labels = clsLabels.clone();
		m_gravF = this.getGravForce(m_model);
	}

	private double[] getAvDistance(ArrayList<Instances> m_model) {
		double[] averageDistanceArr = new double[m_model.size()];
		for (int classID = 0; classID < m_model.size(); classID++) {

			Instances m_instances = new Instances(m_model.get(classID));
			Instance cntrInst = (Instance) this.m_centres[classID].copy();
			double distance = 0;
			for (int i = 0; i < m_instances.numInstances(); i++) {
				distance += CalculateED(cntrInst, m_instances.instance(i));
				// distance= ED.distance(cntrInst, m_instances.get(i));

			}

			averageDistanceArr[classID] = distance / m_instances.numInstances();
		}
		return averageDistanceArr;
	}

	private double getGravForce(ArrayList<Instances> m_model) {
		// TODO Auto-generated method stub
		int k = 0;
		double gravF = 0, temp = 0;
		for (int n = 0; n < this.m_noClasses; n++) {
			for (int i = n + 1; i < this.m_noClasses; i++) {
				temp = gravetationForce(m_model.get(n).numInstances(), m_model.get(i)
						.numInstances(), this.m_centres[n], this.m_centres[i]);
				k++;
				if (temp > gravF)
					gravF = temp;
			}
		}
		// gravF/=k;
		return gravF;
	}

	private Instance[] getSD(ArrayList<Instances> m_model) {
		// TODO Auto-generated method stub
		Instance[] SDInstances = new Instance[this.m_noClasses];
		for (int classID = 0; classID < this.m_noClasses; classID++) {

			Instances m_instances = new Instances(
					removeClass(m_model.get(classID)));
			SDInstances[classID] = (Instance) m_instances.firstInstance()
					.copy();
			double SD[] = new double[m_instances.numAttributes()];
			// Instances microCluster= createMicroCluster(m_instances,classID,
			// 3000);
			for (int i = 0; i < m_instances.numAttributes(); i++)
				SD[i] = m_instances.variance(i);

			for (int j = 0; j < SD.length; j++) {
				SD[j] = Math.sqrt(SD[j]);
				SDInstances[classID].setValue(j, SD[j]);
			}

		}
		// inst.setClassValue(-1);
		return SDInstances;

	}

	private int[] getNoInstaces(ArrayList<Instances> m_model) {
		// TODO Auto-generated method stub
		int[] count = new int[m_noClasses];
		for (int i = 0; i < this.m_noClasses; i++) {
			Instances m_instances = new Instances(m_model.get(i));
			count[i] = m_instances.numInstances();
		}
		return count;
	}

	private Instance[] getCentres(ArrayList<Instances> m_model) {
		// TODO Auto-generated method stub
		Instances new_Instances = new Instances(removeClass(m_model.get(0)));
		Instance[] centres = new Instance[this.m_noClasses];
		// centresArr= new double[m_noClasses][centres.numAttributes()];

		for (int classID = 0; classID < this.m_noClasses; classID++) {

			centres[classID] = (Instance) new_Instances.firstInstance().copy();
			Instances m_instances = new Instances(
					removeClass(m_model.get(classID)));
			int m_numAtt = m_instances.numAttributes();
			double[] mean = new double[m_numAtt];
			for (int i = 0; i < m_numAtt; i++) {
				mean[i] = m_instances.meanOrMode(i);
			}

			for (int i = 0; i < m_numAtt; i++)
				centres[classID].setValue(i, mean[i]);
		}

		return centres;
	}

	private double[] getDensity(ArrayList<Instances> m_model) {

		double[] density = new double[this.m_noClasses];

		for (int classID = 0; classID < this.m_noClasses; classID++) {

			Instances m_instances = new Instances(
					removeClass(m_model.get(classID)));
			m_instances.setClassIndex(-1);

			Instances microCluster = createMicroCluster(m_instances, classID,
					densThreshold);

			EuclideanDistance ED = new EuclideanDistance(microCluster);
			ED.setDontNormalize(true);
			int pSum = 0;
			Instance inst = (Instance) this.m_centres[classID].copy();
			double disSum = 0;

			double counter = 0;
			// double vol= Volume(inst);
			for (int i = 0; i < microCluster.numInstances(); i++) {
				// double d= CalculateED(inst, m_instances.get(i));
				double d = ED.distance(inst, microCluster.instance(i));
				disSum += d;
			}
			counter = microCluster.numInstances();
			// double d= counter/m_instances.size();
			density[classID] = counter;

			// density[classID]= m_instances.size()/avDis;

			// double maxDistance= m_classBoundry[classID];
			// density[classID]= m_instances.size()/maxDistance;
		}

		return density;
	}

	private Instances createMicroCluster(Instances m_instances, int classID,
			double densThreshold) {
		// TODO Auto-generated method stub
		Instances microCluster = new Instances(m_instances, 0);
		Instance inst = (Instance) this.m_centres[classID].copy();
		// EuclideanDistance ED= new EuclideanDistance(m_instances);
		for (int i = 0; i < m_instances.numInstances(); i++) {
			double d = CalculateED(inst, m_instances.instance(i));
			if (d < densThreshold)
				microCluster.add(m_instances.instance(i));

		}

		return microCluster;
	}

	// private double Volume(Instance inst) {
	// // TODO Auto-generated method stub
	// double vol=0;
	// double C=0;
	//
	// int n= inst.numAttributes()-2;
	// double c1= Math.pow(Math.PI, (n/2));
	//
	// return 0;
	// }

	public Model(Instances m_Train, int mode) {

		ArrayList<Instances> m_model = new ArrayList<Instances>();
		// Constructor for labelled data ( assignment mode)
		switch (mode) {
		case 1:
			m_model = buildAssignmentModel(m_Train);

		}

		m_centres = getCentres(m_model);
		m_classBoundry = getClassBoundry(m_model);
		m_averageDisatnce = getAvDistance(m_model);
		m_size = new int[m_noClasses];
		m_size = getNoInstaces(m_model);
		densThreshold = 1000;
		m_SD = getSD(m_model);
		m_sysCapacity = 0;
		for (int j = 0; j < this.m_noClasses; j++) {
			m_sysCapacity += m_model.get(j).numInstances();
		}

		// labels= clsLabels.clone();
		m_gravF = this.getGravForce(m_model);

		m_noClasses = this.getNoClasses();

	}

	private double[] getAverageDistance(ArrayList<Instances> m_model) {
		// TODO Auto-generated method stub
		return this.m_averageDisatnce;
	}

	// private static Model instance;
	//
	// static {
	// instance = new Model();
	// }
	//
	// private Model() {
	// // hidden constructor
	// }
	//
	// public static Model getInstance() {
	// return instance;
	// }

	public Model(Model baseModel) {
		// TODO Auto-generated constructor stub

		this.densThreshold = baseModel.densThreshold;
		this.labels = baseModel.labels.clone();
		this.m_centres = baseModel.m_centres.clone();
		this.m_classBoundry = baseModel.m_classBoundry.clone();
		// this.m_classDensity= baseModel.m_classDensity.clone();
		this.m_averageDisatnce = baseModel.m_averageDisatnce.clone();
		this.m_gravF = baseModel.m_gravF;
		this.m_noClasses = baseModel.m_noClasses;
		this.m_SD = baseModel.m_SD.clone();
		this.m_size = baseModel.m_size.clone();
		this.m_sysCapacity = baseModel.m_sysCapacity;

	}

	protected int getNoClasses() {
		return m_noClasses;
	}

	public void generateModel(Instances m_Train, int mode) {
		ArrayList<Instances> m_model = new ArrayList<Instances>();
		switch (mode) {
		case 1:
			m_model = buildAssignmentModel(m_Train);
			// case 2:
			// buildClusModel(m_Train);
			// case 3:
			// buildClassModel(m_Train);
			// default: buildAssignmentModel(m_Train);

		}

	}

	private ArrayList<Instances> buildAssignmentModel(Instances m_Train) {
		// TODO Auto-generated method stub

		ArrayList<Instances> m_model = new ArrayList<Instances>();
		Instances[] m_modelInsts;

		Attribute classAttr = m_Train.classAttribute();
		m_modelInsts = new Instances[classAttr.numValues()];

		// create attrInfo FastVictor
		FastVector attrInfo = new FastVector();

		for (Enumeration enumAttr = m_Train.enumerateAttributes(); enumAttr
				.hasMoreElements();) {
			attrInfo.addElement((Attribute) enumAttr.nextElement());
		}
		attrInfo.addElement(m_Train.classAttribute());

		// init new instances
		for (int i = 0; i < m_modelInsts.length; i++) {
			m_modelInsts[i] = new Instances(classAttr.value(i), attrInfo, 0);
			m_modelInsts[i].setClassIndex(m_Train.classIndex());
		}

		for (Enumeration e = m_Train.enumerateInstances(); e.hasMoreElements();) {
			Instance current = (Instance) e.nextElement();
			String name = current.stringValue(current.numAttributes() - 1);
			for (int i = 0; i < m_modelInsts.length; i++) {
				if (classAttr.value(i) == name) {

					m_modelInsts[i].add(current);
					break;
				}
			}

		}

		// ArrayList<Instances> modelArr= new ArrayList<Instances>();
		for (int i = 0; i < m_modelInsts.length; i++) {
			if (m_modelInsts[i].numInstances() != 0) {
				m_model.add(m_modelInsts[i]);

			}

		}
		this.labels = new String[m_model.size()];
		for (int j = 0; j < m_model.size(); j++) {
			Instance c = (Instance) m_model.get(j).firstInstance().copy();
			this.labels[j] = c.stringValue(c.numAttributes() - 1);
		}
		this.m_noClasses = m_model.size();

		return m_model;

	}

	public String modelStatistices() throws Exception {
		// TODO Auto-generated method stub
		String Str = "";

		Str += " \n No . of clusters in the model = " + this.getNoClasses()
				+ " \n";
		Str += " Gravitational Force  = " + this.m_gravF + "\n";
		// Str+=" Micro Cluster Size= "+this.densThreshold+"\n";
		for (int i = 0; i < this.getNoClasses(); i++) {
			Str += " No . of points in Cluster :" + i + " Labeled "
					+ this.labels[i] + " = " + this.m_size[i] + "\n";
			Str += " Cluster Centre = " + this.m_centres[i].toString() + "\n"
					+ " Number of attributes " + m_centres[i].numAttributes()
					+ "\n";
			Str += " Cluster Centre Array=  ";
			// for( int n=0; n< this.centresArr[i].length; n++)
			// Str+="  "+ this.centresArr[i][n];
			Str += " Max distance  = " + this.m_classBoundry[i] + "\n";
			Str += " Average Distance  = " + this.m_averageDisatnce[i] + " \n";
			Str += " SD  = " + this.m_SD[i].toString() + "\n";
			Str += " ////////////////////////////////////////////" + "\n";
		}

		return Str;

	}

	private double[] Sum(double[] first, double[] second) throws Exception {
		// TODO Auto-generated method stub
		double[] result = new double[first.length];
		if (first.length != second.length)
			throw new Exception(
					"Instaces are not compatiable : StreamAR Addition");
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] + second[i];
		}
		return result;
	}

	private double[] substarctSQR(double[] first, double[] second)
			throws Exception {
		// TODO Auto-generated method stub
		double[] result = new double[first.length];
		if (first.length != second.length)
			throw new Exception(
					"Instaces are not compatiable : StreamAR Substraction");
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] - second[i];
			result[i] = result[i] * result[i];
		}
		return result;
	}

	protected double[] getClassBoundry(ArrayList<Instances> m_model) {

		double[] maxDistanceArr = new double[m_model.size()];
		for (int classID = 0; classID < m_model.size(); classID++) {

			Instances m_instances = new Instances(m_model.get(classID));
			// EuclideanDistance ED= new EuclideanDistance(m_instances);
			// ED.setDontNormalize(true);
			double maxDistance = 0;
			Instance cntrInst = (Instance) this.m_centres[classID].copy();
			double distance = 0;
			for (int i = 0; i < m_instances.numInstances(); i++) {
				distance = CalculateED(cntrInst, m_instances.instance(i));
				// distance= ED.distance(cntrInst, m_instances.get(i));

				if (distance > maxDistance)
					maxDistance = distance;
			}

			maxDistanceArr[classID] = maxDistance;
		}
		return maxDistanceArr;
	}

	public ArrayList<Prediction> compareModels(Model baseModel)
			throws Exception {

		double[][] densityMap = new double[this.m_noClasses][2];
		int[][] clusterClassMap = new int[this.m_noClasses][baseModel.m_noClasses];

		double temp = 0;
		int classID;
		double avDis = 0;
		double avGrav = 0;
		double avSTD = 0;
		// comparing the models density

		// OLD densityFunction ( Mictrocluster)
		// for( int cluster=0; cluster<this.m_noClasses;cluster++ ){
		// density=Double.POSITIVE_INFINITY;;
		// classID=-1;
		// temp=0;
		// for (int i = 0; i < baseModel.m_noClasses; i++) {
		// temp=
		// Math.abs(baseModel.m_classDensity[i]-this.m_classDensity[cluster]);
		// if( temp< density || temp== density){
		// density= temp;
		// classID= i;
		// }
		// }
		// densityMap[cluster][0]= classID;
		// densityMap[cluster][1]= density;
		// if( classID!=-1)
		// clusterClassMap[cluster][classID]++;
		//
		// }

		// comparing the models density2- Density Gain

		// Get the density gain when the new small cluster jopined any of
		// exiting base model ones.
		// If the NC is Fully contatinedin any base model cluster, The radis
		// stays the same.
		// If the NC is intersects or seperated, The radius is updated,
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {

			classID = -1;
			temp = 0;
			double tempGain = 0;
			double gain = -50000;
			for (int i = 0; i < baseModel.m_noClasses; i++) {
				double newRadius = 0;
				if (inside(this.m_centres[cluster], baseModel.m_centres[i],
						baseModel.m_averageDisatnce[i])
						&& fullContained(this.m_centres[cluster],
								this.m_classBoundry[cluster],
								baseModel.m_centres[i],
								baseModel.m_averageDisatnce[i]))
					newRadius = baseModel.m_averageDisatnce[i];

				else {
					newRadius = this.m_averageDisatnce[cluster]
							+ baseModel.m_averageDisatnce[i]
							+ CalculateED(this.m_centres[cluster],
									baseModel.m_centres[i]);
					newRadius /= 2;
				}

				double oldDensity = densityFunction(
						(double) baseModel.m_size[i],
						baseModel.m_averageDisatnce[i], 3);
				// double oldDensity= densityFunction(baseModel.m_size[i],
				// baseModel.m_classBoundry[i],3);

				int newsize = baseModel.m_size[i] + this.m_size[cluster];
				double newDensity = densityFunction((double) newsize,
						newRadius, 3);
				// double newDensity= densityFunction(newsize, newRadius, 3) ;
				tempGain = newDensity - oldDensity;
				if (tempGain > gain || tempGain == gain) {
					gain = tempGain;
					classID = i;
				}
			}
			densityMap[cluster][0] = classID;
			densityMap[cluster][1] = gain;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;

		}
		// Comparing model centres
		double[][] distanceMap = new double[this.m_noClasses][2];
		classID = -1;

		double distance;

		// EuclideanDistance ED= new EuclideanDistance(this.m_centres);

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			distance = Double.POSITIVE_INFINITY;
			;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {
				Instance baseClsCntr = (Instance) baseModel.m_centres[i].copy();
				temp = CalculateED(clustCntr, baseClsCntr);

				if (temp < distance || temp == distance) {
					distance = temp;
					classID = i;
				}
			}
			distanceMap[cluster][0] = classID;
			distanceMap[cluster][1] = distance;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
			// avDis+=distance;
		}

		//

		// Comparing model SD
		double[][] SDMap = new double[this.m_noClasses][2];
		temp = 0;
		double SD;

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustSD = (Instance) this.m_SD[cluster].copy();
			classID = -1;
			temp = 0;
			SD = Double.POSITIVE_INFINITY;
			for (int i = 0; i < baseModel.m_noClasses; i++) {
				Instance baseClsSD = (Instance) baseModel.m_SD[i].copy();
				temp = CalculateED(clustSD, baseClsSD);
				if (temp < SD || SD == temp) {
					SD = temp;
					classID = i;
				}
			}
			SDMap[cluster][0] = classID;
			SDMap[cluster][1] = SD;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
			avSTD += SD;
		}

		// Comparing model sizes
		// double[][] sizeMap=new double[this.m_noClasses][2];
		// double diff;
		// for( int cluster=0; cluster<this.m_noClasses;cluster++ ){
		// // double clusSize= this.getClassSizePro(cluster);
		// classID=-1;
		// diff=Double.POSITIVE_INFINITY;;
		// for (int i = 0; i < baseModel.m_noClasses; i++) {
		// // double clsSize=baseModel.getClassSizePro(i);
		// temp= Math.abs(clusSize-clsSize);
		// if( temp < diff || temp==diff){
		// diff= temp;
		// classID= i;
		// }
		// }
		// sizeMap[cluster][0]= classID;
		// sizeMap[cluster][1]= diff;
		// // clusterClassMap[cluster][classID]++;
		// }
		//
		//

		// Comparing model gravity
		double[][] gravMap = new double[this.m_noClasses][2];
		temp = 0;
		double gravF;
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clusCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			gravF = 0;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {
				Instance baseClsCntr = (Instance) baseModel.m_centres[i].copy();
				temp = gravetationForce(this.m_size[cluster],
						baseModel.m_size[i], clusCntr, baseClsCntr);
				if (temp > gravF) {
					gravF = temp;
					classID = i;
				}
				avGrav += temp;
			}
			gravMap[cluster][0] = classID;
			gravMap[cluster][1] = gravF;
			clusterClassMap[cluster][classID]++;
			avGrav += gravF;
		}

		// Detecting new clusters

		avDis /= this.m_noClasses;
		avSTD /= this.m_noClasses;

		// average gravitational force between this cluster and all classes in
		// base system
		avGrav /= baseModel.m_noClasses;

		// for( int cluster=0; cluster<this.m_noClasses;cluster++ ){
		// if( gravMap[cluster][1]< avGrav)
		// gravMap[cluster][0]= -1;
		// if( distanceMap[cluster][1]>avDis)
		// distanceMap[cluster][0]= -1;
		// if( SDMap[cluster][1]> avSTD)
		// SDMap[cluster][0]= -1;
		//
		// }

		// the majority votes and displaying result

		String Str = "";

		Str += " Comparing Models \\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n";
		ArrayList<Prediction> predArray = new ArrayList<Prediction>();
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			classID = -1;
			temp = -1;
			Str += " Cluster #" + cluster + "\n============\n";
			Str += " Votes for ClassNo:  ";
			Prediction pd = new Prediction();
			for (int c = 0; c < baseModel.m_noClasses; c++) {
				if (clusterClassMap[cluster][c] > temp) {
					temp = clusterClassMap[cluster][c];
					classID = c;
				}
			}
			if (temp == 2) {
				boolean active = false;
				// Active case
				for (int classNo = 0; classNo < baseModel.m_noClasses; classNo++) {
					if (classNo != classID
							&& clusterClassMap[cluster][classNo] == 2) {

						String[] pLabels = new String[2];
						pLabels[0] = baseModel.labels[classNo].trim()
								.toLowerCase();
						pLabels[1] = baseModel.labels[classID].trim()
								.toLowerCase();
						pd.setActiveParameters(cluster, pLabels,
								this.labels[cluster].trim());
						active = true;
						break;
					}
				}
				if (!active)
					pd.setParameters(cluster, baseModel.labels[classID].trim()
							.toLowerCase(), this.labels[cluster].trim()
							.toLowerCase());

			} else if (temp == 1) {
				// confusion case
				classID = -1;
				pd.setParameters(cluster, "Unrecognised", this.labels[cluster]
						.trim().toLowerCase());

			} else
				pd.setParameters(cluster, baseModel.labels[classID].trim()
						.toLowerCase(), this.labels[cluster].trim()
						.toLowerCase());

			String trueLabel = pd.getTrueLabel();
			boolean denF = false;
			boolean disF = false;
			boolean SDF = false;
			boolean gF = false;
			if (baseModel.labels[(int) densityMap[cluster][0]].trim()
					.toLowerCase().contains(trueLabel.trim()))
				denF = true;
			if (baseModel.labels[(int) distanceMap[cluster][0]].trim()
					.toLowerCase().contains(trueLabel.trim()))
				disF = true;
			if (baseModel.labels[(int) SDMap[cluster][0]].trim().toLowerCase()
					.contains(trueLabel.trim()))
				SDF = true;
			if (baseModel.labels[(int) gravMap[cluster][0]].trim()
					.toLowerCase().contains(trueLabel.trim()))
				gF = true;

			pd.setMeasures(denF, disF, SDF, gF);

			Str += "\n  Prediction Type: " + pd.getType() + " ; "
					+ " Predicted class " + pd.getPredLabel()
					+ " Confidence level:" + pd.getConfLevel() + " True Label "
					+ pd.getTrueLabel();

			Str += "\n Density candidate  " + densityMap[cluster][0]
					+ "   with density diff= " + densityMap[cluster][1];
			Str += "\n Distance candidate  " + distanceMap[cluster][0]
					+ "   with distance diff= " + distanceMap[cluster][1];
			Str += "\n SD candidate  " + SDMap[cluster][0]
					+ "   with SD diff= " + SDMap[cluster][1];
			Str += "\n gravity candidate  " + gravMap[cluster][0]
					+ "   with gravity force diff= " + gravMap[cluster][1];

			pd.setStatatsistics(Str);
			predArray.add(pd);

			Str = " ";
		}
		return predArray;

	}

	public ArrayList<Prediction> compareModels2(BLM baseModel)
			throws Exception {

		double[][] densityMap = new double[this.m_noClasses][3];
		int[][] clusterClassMap = new int[this.m_noClasses][baseModel.m_noClasses];

		double avDis = 0;
		double avGrav = 0;
		double avSTD = 0;
		double temp = 0;
		int classID;
		int subClusterID;

		// Densiy
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {

			classID = -1;
			subClusterID = -1;
			temp = 0;
			double tempGain = 0;
			double gain = -1 * Double.MAX_VALUE;

			for (int i = 0; i < baseModel.m_noClasses; i++) {
				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				
					double newRadius = 0;
					if (inside(this.m_centres[cluster], obj.m_classCentre,
							obj.m_globalAvDistance)
							&& fullContained(this.m_centres[cluster],
									this.m_classBoundry[cluster],
									obj.m_classCentre,
									obj.m_globalAvDistance))
						newRadius = obj.m_globalAvDistance;
					else {
						newRadius = this.m_classBoundry[cluster]
								+ obj.m_globalAvDistance
								+ CalculateED(this.m_centres[cluster],
										obj.m_classCentre);
						newRadius /= 2;
					}
					
					double oldDensity = densityFunction((double) obj.m_totalsize,
							obj.m_globalAvDistance, 3);
					int newsize = obj.m_totalsize + this.m_size[cluster];

					
					double newDensity = densityFunction((double) newsize,
							newRadius, 3);
					tempGain = (newDensity - oldDensity);
					if (tempGain > 0 || tempGain> obj.m_maxVDL) {
						gain = tempGain;
						classID = i;
						
					
				}
			}
//			densityMap[cluster][2] = subClusterID;
			densityMap[cluster][0] = classID;
			densityMap[cluster][1] = gain;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;

		}
		// Comparing model centres
		double[][] distanceMap = new double[this.m_noClasses][3];
		classID = -1;

		double distance;

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			subClusterID = -1;
			distance = Double.POSITIVE_INFINITY;
			;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {
					Instance baseClsCntr = (Instance) obj.m_centres[j].copy();
					temp = CalculateED(clustCntr, baseClsCntr);

					if (temp < distance || temp == distance) {
						distance = temp;
						classID = i;
						subClusterID = j;

					}
				}
			}
			distanceMap[cluster][2] = subClusterID;
			distanceMap[cluster][0] = classID;
			distanceMap[cluster][1] = distance;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
			 avDis+=distance;
		}

		//

		// Comparing model SD
		double[][] SDMap = new double[this.m_noClasses][3];
		temp = 0;
		double SD;

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustSD = (Instance) this.m_SD[cluster].copy();
			classID = -1;
			subClusterID = -1;
			temp = 0;
			SD = Double.POSITIVE_INFINITY;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {

					Instance baseClsSD = (Instance) obj.m_SD[j].copy();
					double mag_clustSD = mag(clustSD);
					double mag_baseClsSD = mag(baseClsSD);
					temp = Math.abs(mag_baseClsSD - mag_clustSD);
					if (temp < SD || SD == temp) {
						SD = temp;
						classID = i;
						subClusterID = j;
					}
				}
			}
			SDMap[cluster][2] = subClusterID;
			SDMap[cluster][0] = classID;
			SDMap[cluster][1] = SD;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
			 avSTD+= SD;
		}

		// Comparing model gravity
		double[][] gravMap = new double[this.m_noClasses][3];
		temp = 0;
		double gravF;
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clusCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			subClusterID = -1;
			gravF = 0;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {
					Instance baseClsCntr = (Instance) obj.m_centres[j].copy();
					temp = gravetationForce(this.m_size[cluster],
							obj.m_size[j], clusCntr, baseClsCntr);
					if (temp > gravF) {
						gravF = temp;
						classID = i;
						subClusterID = j;
					}
					// avGrav+=temp;
				}
			}
			gravMap[cluster][2] = subClusterID;
			gravMap[cluster][0] = classID;
			gravMap[cluster][1] = gravF;
			clusterClassMap[cluster][classID]++;
			 avGrav+= gravF;
		}

		// Detecting new clusters

		avDis /= this.m_noClasses;
		avSTD /= this.m_noClasses;

		avGrav /= baseModel.m_noClasses;

		// the majority votes and displaying result

		String Str = "";

		Str += " Comparing Models \\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n";
		ArrayList<Prediction> predArray = new ArrayList<Prediction>();
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			classID = -1;
			temp = -1;
			Str += "\n Cluster #" + cluster + "\n============\n";
			Str += " Votes for ClassNo:  ";
			Prediction pd = new Prediction();
			pd.setSize(this.m_size[cluster]);
			for (int c = 0; c < baseModel.m_noClasses; c++) {
				if (clusterClassMap[cluster][c] > temp) {
					temp = clusterClassMap[cluster][c];
					classID = c;
				}
			}
			if (temp == 2) {
				boolean active = false;
				// Active case
				for (int classNo = 0; classNo < baseModel.m_noClasses; classNo++) {
					if (classNo != classID
							&& clusterClassMap[cluster][classNo] == 2) {

						String[] pLabels = new String[2];
						pLabels[0] = baseModel.getLabels()[classNo].trim()
								.toLowerCase();
						pLabels[1] = baseModel.getLabels()[classID].trim()
								.toLowerCase();
						pd.setActiveParameters(cluster, pLabels,
								this.labels[cluster].trim());
						active = true;
						break;
					}
				}
				if (!active)
					pd.setParameters(cluster, baseModel.getLabels()[classID]
							.trim().toLowerCase(), this.labels[cluster].trim()
							.toLowerCase());

			} else if (temp == 1) {
				// confusion case
				classID = -1;
				pd.setParameters(cluster, "Unrecognised", this.labels[cluster]
						.trim().toLowerCase());

			} else
				pd.setParameters(cluster, baseModel.getLabels()[classID].trim()
						.toLowerCase(), this.labels[cluster].trim()
						.toLowerCase());

			String trueLabel = pd.getTrueLabel();
			boolean denF = false;
			boolean disF = false;
			boolean SDF = false;
			boolean gF = false;
			if (baseModel.getLabels()[(int) densityMap[cluster][0]]
					.toLowerCase().trim()
					.contains(trueLabel.toLowerCase().trim()))
				denF = true;
			if (baseModel.getLabels()[(int) distanceMap[cluster][0]]
					.toLowerCase().trim()
					.equals(trueLabel.toLowerCase().trim()))
				disF = true;
			if (baseModel.getLabels()[(int) SDMap[cluster][0]].toLowerCase()
					.trim().equals(trueLabel.toLowerCase().trim()))
				SDF = true;
			if (baseModel.getLabels()[(int) gravMap[cluster][0]].toLowerCase()
					.trim().equals(trueLabel.toLowerCase().trim()))
				gF = true;

			pd.setMeasures(denF, disF, SDF, gF);

			Str += "\n  Prediction Type: " + pd.getType() + " ; "
					+ " Predicted class " + pd.getPredLabel()
					+ " Confidence level:" + pd.getConfLevel() + " True Label "
					+ pd.getTrueLabel();

			Str += "\n Density candidate  " + densityMap[cluster][0]
					+ "   with density diff= " + densityMap[cluster][1];
			Str += "\n Distance candidate  " + distanceMap[cluster][0]
					+ "   with distance diff= " + distanceMap[cluster][1];
			Str += "\n SD candidate  " + SDMap[cluster][0]
					+ "   with SD diff= " + SDMap[cluster][1];
			Str += "\n gravity candidate  " + gravMap[cluster][0]
					+ "   with gravity force diff= " + gravMap[cluster][1];

			pd.setStatatsistics(Str);
			predArray.add(pd);

			Str = " ";
		}
		return predArray;

	}

	private double densityFunction(double size, double radius, int d) {

		if (size == 0)
			return -1;

		double V = (4 / 3) * Math.PI * Math.pow(radius, d);

		return size / V;
	}

	private boolean fullContained(Instance c1, double r1, Instance c2, double r2) {
		// TODO Auto-generated method stub
		double d = CalculateED(c1, c2) + r1;
		if (d < r2)
			return true;

		return false;
	}

	private boolean inside(Instance inst1, Instance inst2, double radius2) {
		// TODO Auto-generated method stub
		double d = CalculateED(inst1, inst2);
		if (d < radius2)
			return true;

		return false;
	}

	private double gravetationForce(int sizei, int sizej, Instance m_centres2,
			Instance m_centres3) {
		// TODO Auto-generated method stub
		double distance;
		// EuclideanDistance ED= new EuclideanDistance(this.m_centres);
		distance = CalculateED(m_centres2, m_centres3);
		double result = (sizei * sizej) / (distance * distance);
		return result;
	}

	private Instances removeClass(Instances inst) {
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

	//
	// private double getClassSizePro(int i) {
	// // TODO Auto-generated method stub
	//
	// double value=(double)this.m_model.get(i).size()/m_sysCapacity ;
	// return value;
	// }

	private double CalculateED(Instance first, Instance second) {

		double dist = 0.0;

		for (int i = 0; i < first.numAttributes(); i++) {
			double x = first.value(i);
			double y = second.value(i);

			if (Double.isNaN(x) || Double.isNaN(y)) {
				continue; // Mark missing attributes ('?') as NaN.
			}

			dist += (x - y) * (x - y);
		}

		return Math.sqrt(dist);
	}

	protected String[] modelLabels() {
		return labels;

	}

	protected String predection(int modelClustNo) {
		if (modelClustNo < predection.length)
			return predection[modelClustNo];
		return " ";
	}

	protected int predictionStatus(int modelClusNo) {
		if (modelClusNo < predStatus.length)
			return predStatus[modelClusNo];
		else
			return -1;

	}

	public String updateModel(Instances m_instances, String label) {

		String s = "";
		int clusID = -1;
		for (int i = 0; i < this.labels.length; i++) {
			if (labels[i].equals(label)) {
				clusID = i;
				break;
			}
		}

		if (clusID != -1) {
			Instances newInst = new Instances(removeClass(m_instances));

			Instance oldCentre = (Instance) this.m_centres[clusID].copy();
			this.m_centres[clusID] = updateCentre(clusID, newInst, oldCentre);

			double centreMov = CalculateED(oldCentre, this.m_centres[clusID]);

			double oldBoundry = this.m_classBoundry[clusID];
			this.m_classBoundry[clusID] = updateBoundry(clusID, newInst,
					oldBoundry);

			int oldSize = this.m_size[clusID];
			this.m_size[clusID] += newInst.numInstances();

			double oldDensityThreshold = densThreshold;
			this.densThreshold = UpdateDensThreshold();

			// double oldDensity= this.m_classDensity[clusID];
			// double mcSize= oldDensity*oldSize;
			// this.m_classDensity[clusID]= updateDensity(clusID,
			// newInst,mcSize);

			Instance oldSD = (Instance) this.m_SD[clusID].copy();
			this.m_SD[clusID] = updateSD(clusID, newInst, oldSD, oldCentre,
					oldSize);

			double SDDiff = CalculateED(oldSD, this.m_SD[clusID]);

			double oldCapacity = m_sysCapacity;
			this.m_sysCapacity += newInst.numInstances();

			double oldGravForce = this.m_gravF;
			this.m_gravF = UpdateGravForce();

			s = "\n*********************************\nModel Updated: \n"
					+ " Centre Movement= "
					+ centreMov
					+ "\n"
					+ " SD Diffrence= "
					+ SDDiff
					+ "\n"
					+ " OldBoundries= "
					+ oldBoundry
					+ " New Boundries= "
					+ this.m_classBoundry[clusID]
					+ "\n"
					+ " Old Size= "
					+ oldSize
					+ " New Size= "
					+ this.m_size[clusID]
					+ "\n"
					// +" Old Density= "+ oldDensity+" New Density= "+
					// this.m_classDensity[clusID]+"\n"
					+ " Old Capacity= "
					+ oldCapacity
					+ " New Capacity= "
					+ this.m_sysCapacity
					+ "\n"
					+ " Old Gravity= "
					+ oldGravForce
					+ " New Gravity= "
					+ this.m_gravF
					+ "\n"
					+ " Density Threshold= "
					+ oldDensityThreshold
					+ " New Threshold= " + this.densThreshold + "\n";
			;
		} else {
			s = " Failed to update";
		}

		return s;

	}

	public String updateCSWSCModel(Instances m_instances, String label) {

		String s = "";
		int clusID = -1;
		for (int i = 0; i < this.labels.length; i++) {
			if (labels[i].equals(label)) {
				clusID = i;
				break;
			}
		}

		if (clusID != -1) {
			Instances newInst = new Instances(removeClass(m_instances));

			Instance oldCentre = (Instance) this.m_centres[clusID].copy();
			this.m_centres[clusID] = updateCentre(clusID, newInst, oldCentre);

			double centreMov = CalculateED(oldCentre, this.m_centres[clusID]);

			double oldBoundry = this.m_classBoundry[clusID];
			this.m_classBoundry[clusID] = updateBoundry(clusID, newInst,
					oldBoundry);

			int oldSize = this.m_size[clusID];
			this.m_size[clusID] += newInst.numInstances();

			double oldDensityThreshold = densThreshold;
			this.densThreshold = UpdateDensThreshold();

			// double oldDensity= this.m_classDensity[clusID];
			// double mcSize= oldDensity*oldSize;
			// this.m_classDensity[clusID]= updateDensity(clusID,
			// newInst,mcSize);

			Instance oldSD = (Instance) this.m_SD[clusID].copy();
			this.m_SD[clusID] = updateSD(clusID, newInst, oldSD, oldCentre,
					oldSize);

			double SDDiff = CalculateED(oldSD, this.m_SD[clusID]);

			double oldCapacity = m_sysCapacity;
			this.m_sysCapacity += newInst.numInstances();

			double oldGravForce = this.m_gravF;
			this.m_gravF = UpdateGravForce();

			s = "\n*********************************\nModel Updated: \n"
					+ " Centre Movement= "
					+ centreMov
					+ "\n"
					+ " SD Diffrence= "
					+ SDDiff
					+ "\n"
					+ " OldBoundries= "
					+ oldBoundry
					+ " New Boundries= "
					+ this.m_classBoundry[clusID]
					+ "\n"
					+ " Old Size= "
					+ oldSize
					+ " New Size= "
					+ this.m_size[clusID]
					+ "\n"
					// +" Old Density= "+ oldDensity+" New Density= "+
					// this.m_classDensity[clusID]+"\n"
					+ " Old Capacity= "
					+ oldCapacity
					+ " New Capacity= "
					+ this.m_sysCapacity
					+ "\n"
					+ " Old Gravity= "
					+ oldGravForce
					+ " New Gravity= "
					+ this.m_gravF
					+ "\n"
					+ " Density Threshold= "
					+ oldDensityThreshold
					+ " New Threshold= " + this.densThreshold + "\n";
			;
		} else {
			s = " Failed to update";
		}

		return s;

	}

	private double UpdateDensThreshold() {

		double densTh = m_classBoundry[0];
		for (int i = 1; i < m_noClasses; i++) {
			if (densTh < m_classBoundry[i])
				densTh = m_classBoundry[i];
		}
		return densTh;
	}

	private double UpdateGravForce() {

		double gravF = 0, temp = 0;
		for (int n = 0; n < this.m_noClasses; n++) {
			for (int i = n + 1; i < this.m_noClasses; i++) {
				temp = gravetationForce(this.m_size[n], this.m_size[i],
						this.m_centres[n], this.m_centres[i]);
				if (temp > gravF)
					gravF = temp;
			}
		}
		return gravF;

	}

	private Instance updateSD(int clusID, Instances newInst, Instance prevSD,
			Instance oldCentre, int oldSize) {

		Instance updatedMean = (Instance) oldCentre.copy();
		int n = oldSize;
		Instance nIns = (Instance) newInst.firstInstance().copy();
		double[] mean = new double[nIns.numAttributes()];
		double[] M2 = new double[nIns.numAttributes()];
		Instance updatedSD = (Instance) prevSD.copy();
		for (int i = 0; i < M2.length; i++) {
			M2[i] = prevSD.value(i) * prevSD.value(i) * n;
		}

		for (int i = 0; i < newInst.numInstances(); i++) {
			n++;
			nIns = newInst.instance(i);
			for (int j = 0; j < nIns.numAttributes(); j++) {
				mean[j] = 0;
				double delta = nIns.value(j) - updatedMean.value(j);
				mean[j] = updatedMean.value(j) + (delta / n);
				M2[j] += delta * (nIns.value(j) - mean[j]);
				updatedMean.setValue(j, mean[j]);
			}

		}
		for (int i = 0; i < M2.length; i++) {
			M2[i] = M2[i] / n;

			updatedSD.setValue(i, Math.sqrt(M2[i]));
		}

		return updatedSD;

	}

	private double updateDensity(int clusID, Instances m_instances,
			double mcSize) {
		// TODO Auto-generated method stub
		Instances newInst = new Instances(removeClass(m_instances));
		newInst.setClassIndex(-1);
		Instance inst = (Instance) this.m_centres[clusID].copy();

		double counter = 0;
		for (int i = 0; i < newInst.numInstances(); i++) {
			double d = CalculateED(newInst.instance(i), inst);
			if (d > densThreshold)
				counter++;
		}

		double d = (mcSize + counter) / densThreshold;
		return d;
	}

	private double updateBoundry(int clusID, Instances newInst,
			double oldBoundry) {
		// TODO Auto-generated method stub
		Instances m_instances = new Instances(removeClass(newInst));
		double maxDistance = oldBoundry;
		Instance newCntr = (Instance) this.m_centres[clusID].copy();
		double distance = 0;
		for (int i = 0; i < m_instances.numInstances(); i++) {

			distance = CalculateED(newCntr, m_instances.instance(i));
			if (distance > maxDistance)
				maxDistance = distance;
		}
		return maxDistance;
	}

	private Instance updateCentre(int clusID, Instances newInst,
			Instance oldCentre) {
		// TODO Auto-generated method stub

		Instance updatedMean = (Instance) oldCentre.copy();
		int n = this.m_size[clusID];
		Instance nIns = (Instance) newInst.firstInstance().copy();
		double[] mean = new double[nIns.numAttributes()];
		for (int i = 0; i < newInst.numInstances(); i++) {
			n++;
			nIns = newInst.instance(i);
			for (int j = 0; j < nIns.numAttributes(); j++) {
				mean[j] = 0;
				double delta = nIns.value(j) - updatedMean.value(j);
				mean[j] = updatedMean.value(j) + (delta / n);
				updatedMean.setValue(j, mean[j]);
			}
		}
		return updatedMean;
	}

	public int[] getSize() {
		return m_size;
	}

	public int getNumClasse() {
		// TODO Auto-generated method stub
		return m_noClasses;
	}

	public ArrayList<Prediction> compareModels(BLM baseModel)
			throws Exception {

		double[][] densityMap = new double[this.m_noClasses][3];
		int[][] clusterClassMap = new int[this.m_noClasses][baseModel.m_noClasses];

		double avDis = 0;
		double avGrav = 0;
		double avSTD = 0;
		double temp = 0;
		int classID;
		int subClusterID;
		boolean distanceflag=false; 

		// Densiy 
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {

			classID = -1;
			subClusterID = -1;
			temp = 0;
			double tempGain = 0;
			double gain = -1 * Double.MAX_VALUE;

			for (int i = 0; i < baseModel.m_noClasses; i++) {
				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for( int j=0; j< baseModel.m_classesWithClusters.get(i).NoOfSubClusters;j++){
					double newRadius = 0;
					if (inside(this.m_centres[cluster], obj.m_centres[j],
							obj.m_classBoundry[j])
							&& fullContained(this.m_centres[cluster],
									this.m_classBoundry[cluster],
									obj.m_centres[j],
									obj.m_classBoundry[j]))
						newRadius = obj.m_classBoundry[j];
					else {
						newRadius = this.m_classBoundry[cluster]
								+ obj.m_classBoundry[j]
								+ CalculateED(this.m_centres[cluster],
										obj.m_centres[j]);
						newRadius /= 2;
					}
					
					double oldDensity = densityFunction((double) obj.m_size[j],
							obj.m_classBoundry[j], 3);
					int newsize = obj.m_size[j] + this.m_size[cluster];
					double newDensity = densityFunction((double) newsize,
							newRadius, 3);
					tempGain = (newDensity - oldDensity);
					if (tempGain > gain || tempGain==gain) {
						gain = tempGain;
						classID = i;
						subClusterID=j; 
					
				}
				}
			}
			densityMap[cluster][2] = subClusterID;
			densityMap[cluster][0] = classID;
			densityMap[cluster][1] = gain;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
		}
		// Comparing model centres
		double[][] distanceMap = new double[this.m_noClasses][3];
		classID = -1;

		double distance;

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			subClusterID = -1;
			distance = Double.POSITIVE_INFINITY;
			;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {
					Instance baseClsCntr = (Instance) obj.m_centres[j].copy();
					temp = CalculateED(clustCntr, baseClsCntr);

					if (temp < distance || temp == distance) {
						distance = temp;
						classID = i;
						subClusterID = j;

					}
				}
			}
			distanceMap[cluster][2] = subClusterID;
			distanceMap[cluster][0] = classID;
			distanceMap[cluster][1] = distance;
			if (classID != -1){
				clusterClassMap[cluster][classID]++;
				distanceflag=true; 
			}
			 avDis+=distance;
		}

		//

		// Comparing model SD
		double[][] SDMap = new double[this.m_noClasses][3];
		temp = 0;
		double SD;

		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clustSD = (Instance) this.m_SD[cluster].copy();
			classID = -1;
			subClusterID = -1;
			temp = 0;
			SD = Double.POSITIVE_INFINITY;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {

					Instance baseClsSD = (Instance) obj.m_SD[j].copy();
					double mag_clustSD = mag(clustSD);
					double mag_baseClsSD = mag(baseClsSD);
					temp = Math.abs(mag_baseClsSD - mag_clustSD);
					if (temp < SD || SD == temp) {
						SD = temp;
						classID = i;
						subClusterID = j;
					}
				}
			}
			SDMap[cluster][2] = subClusterID;
			SDMap[cluster][0] = classID;
			SDMap[cluster][1] = SD;
			if (classID != -1)
				clusterClassMap[cluster][classID]++;
			 avSTD+= SD;
		}

		// Comparing model gravity
		double[][] gravMap = new double[this.m_noClasses][3];
		temp = 0;
		double gravF;
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			Instance clusCntr = (Instance) this.m_centres[cluster].copy();
			classID = -1;
			subClusterID = -1;
			gravF = 0;
			temp = 0;
			for (int i = 0; i < baseModel.m_noClasses; i++) {

				ClassWSubClusters obj = baseModel.m_classesWithClusters.get(i);
				for (int j = 0; j < obj.NoOfSubClusters; j++) {
					Instance baseClsCntr = (Instance) obj.m_centres[j].copy();
					temp = gravetationForce(this.m_size[cluster],
							obj.m_size[j], clusCntr, baseClsCntr);
					if (temp > gravF) {
						gravF = temp;
						classID = i;
						subClusterID = j;
					}
					// avGrav+=temp;
				}
			}
			gravMap[cluster][2] = subClusterID;
			gravMap[cluster][0] = classID;
			gravMap[cluster][1] = gravF;
			clusterClassMap[cluster][classID]++;
			 avGrav+= gravF;
		}

		// Detecting new clusters

		avDis /= this.m_noClasses;
		avSTD /= this.m_noClasses;

		avGrav /= baseModel.m_noClasses;

		// the majority votes and displaying result

		String Str = "";

		Str += " Comparing Models \\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n";
		ArrayList<Prediction> predArray = new ArrayList<Prediction>();
		for (int cluster = 0; cluster < this.m_noClasses; cluster++) {
			classID = -1;
			temp = -1;
			Str += "\n Cluster #" + cluster + "\n============\n";
			Str += " Votes for ClassNo:  ";
			Prediction pd = new Prediction();
			pd.setSize(this.m_size[cluster]);
			for (int c = 0; c < baseModel.m_noClasses; c++) {
				if (clusterClassMap[cluster][c] > temp) {
					temp = clusterClassMap[cluster][c];
					classID = c;
				}
			}
			if (temp == 2) {
				boolean active = false;
				// Active case
				for (int classNo = 0; classNo < baseModel.m_noClasses; classNo++) {
					if (classNo != classID
							&& clusterClassMap[cluster][classNo] == 2) {
						//Go for distance candidate
//						pd.setParameters(cluster, baseModel.getLabels()[(int) distanceMap[cluster][0]].trim()
//								.toLowerCase(), this.labels[cluster].trim()
//								.toLowerCase());
//						// typical Active case
						String[] pLabels = new String[2];
						pLabels[0] = baseModel.getLabels()[classNo].trim()
								.toLowerCase();
						pLabels[1] = baseModel.getLabels()[classID].trim()
								.toLowerCase();
						pd.setActiveParameters(cluster, pLabels,
								this.labels[cluster].trim());
						active = true;
						break;
						
					}
				}
				if (!active)
					pd.setParameters(cluster, baseModel.getLabels()[classID]
							.trim().toLowerCase(), this.labels[cluster].trim()
							.toLowerCase());

			} else if (temp == 1) {
				// confusion case
				classID = -1;
				pd.setParameters(cluster, "Unrecognised", this.labels[cluster]
						.trim().toLowerCase());

			} else
				pd.setParameters(cluster, baseModel.getLabels()[classID].trim()
						.toLowerCase(), this.labels[cluster].trim()
						.toLowerCase());

			String trueLabel = pd.getTrueLabel();
			boolean denF = false;
			boolean disF = false;
			boolean SDF = false;
			boolean gF = false;
			if (baseModel.getLabels()[(int) densityMap[cluster][0]]
					.toLowerCase().trim()
					.contains(trueLabel.toLowerCase().trim()))
				denF = true;
			if (baseModel.getLabels()[(int) distanceMap[cluster][0]]
					.toLowerCase().trim()
					.equals(trueLabel.toLowerCase().trim()))
				disF = true;
			if (baseModel.getLabels()[(int) SDMap[cluster][0]].toLowerCase()
					.trim().equals(trueLabel.toLowerCase().trim()))
				SDF = true;
			if (baseModel.getLabels()[(int) gravMap[cluster][0]].toLowerCase()
					.trim().equals(trueLabel.toLowerCase().trim()))
				gF = true;

			pd.setMeasures(denF, disF, SDF, gF);

			Str += "\n  Prediction Type: " + pd.getType() + " ; "
					+ " Predicted class " + pd.getPredLabel()
					+ " Confidence level:" + pd.getConfLevel() + " True Label "
					+ pd.getTrueLabel();

			Str += "\n Density candidate  " + densityMap[cluster][0] +"   Sub Cluster : "+ densityMap[cluster][2]
					+ "   with density diff= " + densityMap[cluster][1];
			Str += "\n Distance candidate  " + distanceMap[cluster][0]
					+ "   with distance diff= " + distanceMap[cluster][1];
			Str += "\n SD candidate  " + SDMap[cluster][0]
					+ "   with SD diff= " + SDMap[cluster][1];
			Str += "\n gravity candidate  " + gravMap[cluster][0]
					+ "   with gravity force diff= " + gravMap[cluster][1];

			
			
			pd.setStatatsistics(Str);
			predArray.add(pd);

			Str = " ";
		}
		
		return predArray;

	}

	private double mag(Instance ins) {
		double mag = 0;
		for (int i = 0; i < ins.numAttributes(); i++) {
			mag += Math.pow(ins.value(i), 2);

		}
		mag = Math.sqrt(mag);
		return mag;
	}

	// public Model copy(Model baseModel) {
	// // TODO Auto-generated method stub
	// Model m= new Model();
	// m.centresArr= baseModel.centresArr;
	// m.densThreshold= baseModel.densThreshold;
	// m.labels= baseModel.labels;
	// m.m_centres= baseModel.m_centres;
	// m.m_classBoundry= baseModel.m_classBoundry;
	// m.m_classDensity= baseModel.m_classDensity;
	// m.m_gravF= baseModel.m_gravF;
	// m.m_noClasses= baseModel.m_noClasses;
	// m.m_SD= baseModel.m_SD;
	// m.m_size= baseModel.m_size;
	// m.m_sysCapacity= baseModel.m_sysCapacity;
	// return m;
	// }

}