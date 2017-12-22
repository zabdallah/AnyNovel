package StreamAR;

import java.awt.Component;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;

import javax.swing.JOptionPane;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class BLM implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4759763813091459788L;
	public int m_noClasses;
	public double m_sysCapacity;
	public String[] m_labels;
	public String[] m_Oldlabels;
	public String predection[];
	public int predStatus[];
	public ArrayList<ClassWSubClusters> m_classesWithClusters;
	public Instance m_SysCentre;
	public double m_FathestDis;
	public Instance novelClassCentre;
	public Instances suspNovelInstances;
	public String m_predLabel;
	public int[][] casesCounter = new int[5][6];
	public double[] d_max;
	public double oldNoveldensity = 0;
	int[] ALcases = new int[3];
	int[] unknownCases = new int[4];
	boolean unkFlag = false;

	public BLM(ArrayList<ClassWSubClusters> m_model, int NoOFClasses, String[] Labels)

	{
		this.m_classesWithClusters = new ArrayList<ClassWSubClusters>(m_model.size());
		for (int j = 0; j < m_model.size(); j++) {
			ClassWSubClusters obj = m_model.get(j).deepCopy();
			this.m_classesWithClusters.add((ClassWSubClusters) obj);
		}
		this.m_noClasses = NoOFClasses;
		this.m_labels = Labels.clone();
		this.m_Oldlabels = Labels.clone();

		this.m_sysCapacity = 0;
		for (int j = 0; j < this.m_noClasses; j++) {
			this.m_sysCapacity += m_model.get(j).getCapacity();
		}
		this.m_SysCentre = getSysCentre(this.m_classesWithClusters);
		// this.m_FathestDis= getFathestDistance(this.m_classesWithClusters);

		// Novel observation
		this.novelClassCentre = null;

		this.d_max = new double[this.m_noClasses];
		this.d_max = findMaxDistance();
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 6; j++)
				this.casesCounter[i][j] = 0;

	}

	private double[] findMaxDistance() {
		double[] maxDis = new double[this.m_noClasses];
		for (int i = 0; i < this.m_noClasses; i++) {
			double temp = 0, dis = 0;
			ClassWSubClusters obj = this.m_classesWithClusters.get(i);
			for (int j = 0; j < obj.NoOfSubClusters; j++) {
				temp = CalculateED(obj.m_centres[j], obj.m_classCentre);
				if (temp > dis)
					dis = temp;
			}
			maxDis[i] = temp;
		}
		return maxDis;
	}

	private double findNovelMaxDistance() {
		double temp = 0, dis = 0;
		ClassWSubClusters obj = this.m_classesWithClusters.get(this.m_noClasses - 1);
		for (int j = 0; j < obj.NoOfSubClusters; j++) {
			temp = CalculateED(obj.m_centres[j], obj.m_classCentre);
			if (temp > dis)
				dis = temp;
		}
		return temp;
	}

	private Instance getSysCentre(ArrayList<ClassWSubClusters> clusters) {
		Instance middle = (Instance) clusters.get(0).m_classCentre.copy();
		double[] middleArray = new double[middle.numAttributes()];
		for (int i = 0; i < middle.numAttributes(); i++) {
			for (int j = 0; j < this.m_noClasses; j++)
				middleArray[i] += clusters.get(j).m_classCentre.value(i) * clusters.get(j).m_totalsize;
			middleArray[i] /= (double) this.m_sysCapacity;

		}
		for (int i = 0; i < middle.numAttributes(); i++) {
			middle.setValue(i, middleArray[i]);
		}
		return middle;

	}

	@SuppressWarnings("unchecked")
	public BLM(BLM baseModel) {
		// TODO Auto-generated constructor stub

		this.m_labels = baseModel.m_labels.clone();
		this.m_Oldlabels = baseModel.m_Oldlabels.clone();
		this.m_classesWithClusters = new ArrayList<ClassWSubClusters>(baseModel.m_classesWithClusters.size());
		for (int j = 0; j < baseModel.m_classesWithClusters.size(); j++) {
			ClassWSubClusters obj = (ClassWSubClusters) baseModel.m_classesWithClusters.get(j).deepCopy();
			this.m_classesWithClusters.add((ClassWSubClusters) obj);
		}
		this.m_noClasses = baseModel.m_noClasses;
		this.m_sysCapacity = baseModel.m_sysCapacity;
		this.m_FathestDis = baseModel.m_FathestDis;
		this.m_SysCentre = (Instance) baseModel.m_SysCentre.copy();
		this.novelClassCentre = null;
		this.d_max = new double[this.m_noClasses];
		this.d_max = baseModel.d_max;
		this.bufferSizeF = baseModel.bufferSizeF;
		this.JPLayerF = baseModel.JPLayerF;
		this.awaytemp = baseModel.awaytemp;
		this.centreMovement = baseModel.centreMovement;
		this.driftClassID = (ArrayList<Integer>) baseModel.driftClassID.clone();
		this.faNovelFlag = baseModel.faNovelFlag;
		this.reNovelFlag = baseModel.reNovelFlag;
		this.ALcases = baseModel.ALcases.clone();
		this.slackSize = baseModel.slackSize;
		this.stableSize = baseModel.stableSize;
		this.ALStats = baseModel.ALStats;

	}

	private double gravetationForce(int sizei, int sizej, Instance m_centres2, Instance m_centres3) {

		double distance = CalculateED(m_centres2, m_centres3);
		double result = (double) (sizei * sizej) / (distance * distance);
		return result;
	}

	private double densityFunction(double size, double radius, int d) {
		// TODO Auto-generated method stub

		if (size == 0)
			return -1;
		double v = (4 / 3) * Math.PI * Math.pow(radius, d);
		return size / v;
	}

	private boolean fullContained(Instance c1, double r1, Instance c2, double r2) {

		// TODO Auto-generated method stub
		double d = CalculateED(c1, c2) + r1;
		if (d < r2)
			return true;

		return false;
	}

	private boolean inside(Instance inst1, Instance inst2, double radius2) {
		double d = CalculateED(inst1, inst2);
		if (d < radius2)
			return true;

		return false;
	}

	private double CalculateED(Instance first, Instance second) {

		Component frame = null;

		if (first.numAttributes() != second.numAttributes())
			JOptionPane.showMessageDialog(frame, "Eggs are not supposed to be green.");

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

	public String modelStatistices() throws Exception {
		// TODO Auto-generated method stub
		String Str = "";

		Str += " \n No . of classes in the model = " + this.m_noClasses + " \n";

		for (int i = 0; i < this.m_noClasses; i++) {
			ClassWSubClusters obj = this.m_classesWithClusters.get(i);
			Str += " No . of sub clusters in Class :" + i + " is: " + obj.NoOfSubClusters + "\n";
			Str += "		d_max= " + this.d_max[i] + "\n";
			Str += "		max VDL = " + obj.m_maxVDL + "\n";
			Str += "      Min Grav. Force= " + obj.m_minGravF + "\n";
			for (int j = 0; j < obj.NoOfSubClusters; j++) {
				Str += "..........\n";
				Str += " 		No . of points in sub cluster :" + j + " Labeled " + this.m_labels[i] + " = "
						+ obj.m_size[j] + "\n";
				Str += " 		Sub Cluster Centre = " + obj.m_centres[j].toString() + "\n"
						+ " 		Number of attributes " + obj.m_centres[j].numAttributes() + "\n";
				// Str+=" Cluster Centre Array= ";
				// // for( int n=0; n< this.centresArr[i].length; n++)
				// // Str+=" "+ this.centresArr[i][n];
				Str += " 		Max distance  = " + obj.m_classBoundry[j] + "\n";
				Str += "		Average Distance = " + obj.m_averageDistance[j] + "\n";
				Str += " 		SD  = " + obj.m_SD[j].toString() + "\n";

				Str += "		VDL = " + obj.m_VDL[j] + "\n";
				for (int k = 0; k < obj.NoOfSubClusters; k++) {
					Str += "		GF with subcluster " + k + " = " + obj.m_gravForce[j][k] + "\n";
				}
				Str += " ////////////////////////////////////////////" + "\n";

			}
		}

		return Str;

	}

	public Instances getDeclaredData() {
		Instances DD = new Instances(dataWarehouse);
		return DD;
	}

	public boolean isNovelClassCreated() {
		if (this.m_Oldlabels.length == this.m_labels.length)
			return false;
		return true;
	}

	public String updateCSWSCModel(Instances nInst, String label) throws Exception {

		String s = "";
		int pos = -1;
		int classID = -1;
		for (int i = 0; i < this.m_Oldlabels.length; i++) {
			if (this.m_Oldlabels[i].trim().toLowerCase().contains(label.toLowerCase().trim())) {
				classID = i;
				break;
			}
		}

		if (classID != -1) {
			unknownCases[1]++;
			// get subclusters of the class, then choose the subcluster to
			// update
			Instances newInst = new Instances(removeClass(nInst));
			ClassWSubClusters obj = this.m_classesWithClusters.get(classID);
			pos = chooseSubClusterToUpdate(obj, newInst);
			s += " \n ************\n Model Update: SubCluster No:" + pos + " in class: " + classID + " labeled: "
					+ label + "\n";

			// Update local subclusters characteristics

			Instance oldCentre = (Instance) obj.m_centres[pos].copy();
			obj.m_centres[pos] = updateCentre(obj.m_size[pos], newInst, oldCentre);
			double centreMov = CalculateED(oldCentre, obj.m_centres[pos]);
			double oldBoundry = obj.m_classBoundry[pos];
			// double ShrinkedBoundry = obj.m_classBoundry[pos]-centreMov ;
			double boundry = getBoundry(newInst);
			double w = newInst.numInstances() / obj.m_size[pos];
			// double w=1;
			if (inside(getInstancesCentre(newInst), oldCentre, oldBoundry)
					&& fullContained(getInstancesCentre(newInst), boundry, oldCentre, oldBoundry))
				obj.m_classBoundry[pos] = oldBoundry - (w * centreMov);
			else
				obj.m_classBoundry[pos] = oldBoundry + (w * centreMov);
			// obj.m_classBoundry[pos] = updateBoundry(
			// (Instance) obj.m_centres[pos].copy(), newInst, oldBoundry);
			// if( obj.m_classBoundry[pos]== oldBoundry)
			// obj.m_classBoundry[pos]= ShrinkedBoundry;
			int oldSize = obj.m_size[pos];
			obj.m_size[pos] += newInst.numInstances();

			// ////////

			// Update global characteristics
			obj.m_classCentre = updateGlobalCentre(obj.m_centres.clone());
			obj.m_globalBoundry = updateGlobalBoundry(obj.m_classBoundry, obj.m_centres.clone(),
					(Instance) obj.m_classCentre.copy());
			double oldAvDist = obj.m_averageDistance[pos];
			obj.m_averageDistance[pos] = updateAvDistance((Instance) obj.m_centres[pos].copy(), newInst, oldAvDist,
					(double) obj.m_size[pos]);
			obj.m_globalAvDistance = updatedGlobalAvDistance(obj.m_averageDistance);
			obj.m_totalsize += newInst.numInstances();

			Instance oldSD = (Instance) obj.m_SD[pos].copy();
			obj.m_SD[pos] = updateSD(newInst, oldSD, oldCentre, oldSize);
			double SDDiff = CalculateED(oldSD, obj.m_SD[pos]);
			double oldCapacity = m_sysCapacity;
			this.m_sysCapacity += newInst.numInstances();
			double old_FarthestDis = obj.m_farthest_dis;
			// obj.m_farthest_dis=
			// updateFarthestDistance(obj.m_centres.clone());
			obj.m_VDL = updateVDL(obj);
			obj.m_maxVDL = getMaxLoss(obj.m_VDL);
			double oldGF = obj.m_minGravF;
			obj.m_minGravF = sysMinGF(classID);
			double old_d_max[] = new double[this.d_max.length];
			old_d_max = this.d_max.clone();
			this.d_max = this.findMaxDistance();
			s += " Centre Movement= " + centreMov + "\n" + " SD Diffrence= " + SDDiff + "\n" + " OldBoundries= "
					+ oldBoundry + " New Boundries= " + obj.m_classBoundry[pos] + "\n" + " Old Size= " + oldSize
					+ " New Size= " + obj.m_size[pos] + "\n" + " OldAverageDis= " + oldAvDist + " New Average= "
					+ obj.m_averageDistance[pos] + "\n" + " Old Capacity= " + oldCapacity + " New Capacity= "
					+ this.m_sysCapacity + "\n";
			// +" Old Farthest Distance= "+
			// old_FarthestDis+" New= "+obj.m_farthest_dis+"\n"
			// +"Old GF= "+ oldGF+" Updated GF= "+ obj.m_minGravF+"\n";
			// s += " Centre Movement= " + centreMov + "\n";
			for (int n = 0; n < d_max.length; n++) {
				s += " D_max[" + n + "] = " + d_max[n] + "   old: " + old_d_max[n] + "\n";
			}

		} else {

			s += updateModelFn(label);

		}

		return s;

	}

	// public String newUpdateCSWSCModel(Instances nInst, String label)
	// throws Exception {
	//
	// String s = "";
	// int pos = -1;
	// int classID = -1;
	// for (int i = 0; i < this.m_labels.length; i++) {
	// if (this.m_labels[i].trim().toLowerCase()
	// .contains(label.toLowerCase().trim())) {
	// classID = i;
	// break;
	// }
	// }
	//
	// if (classID != -1) {
	// // get subclusters of the class, then choose the subcluster to
	// // update
	// Instances newInst = new Instances(removeClass(nInst));
	// ClassWSubClusters obj = this.m_classesWithClusters.get(classID);
	// pos = chooseSubClusterToUpdate(obj, newInst);
	// s += " \n ************\n Model Update: SubCluster No:" + pos
	// + " in class: " + classID + " labeled: " + label + "\n";
	//
	// // Update local subclusters characteristics
	//
	// Instance oldCentre = (Instance) obj.m_centres[pos].copy();
	// obj.m_centres[pos] = newUpdateCentre(obj.m_size[pos], newInst,
	// oldCentre);
	// double centreMov = CalculateED(oldCentre, obj.m_centres[pos]);
	// double oldBoundry = obj.m_classBoundry[pos];
	// obj.m_classBoundry[pos] = updateBoundry(
	// (Instance) obj.m_centres[pos].copy(), newInst, oldBoundry);
	// int oldSize = obj.m_size[pos];
	// obj.m_size[pos] += newInst.size();
	//
	// // ////////
	//
	// // Update global characteristics
	// double oldAvDist = obj.m_averageDistance[pos];
	// obj.m_averageDistance[pos] = updateAvDistance(
	// (Instance) obj.m_centres[pos].copy(), newInst, oldAvDist,
	// (double)obj.m_size[pos]);
	// obj.m_globalAvDistance = updatedGlobalAvDistance(obj.m_averageDistance);
	// obj.m_classCentre = updateGlobalCentre(obj.m_centres.clone());
	// obj.m_globalBoundry = updateGlobalBoundry(obj.m_classBoundry,
	// obj.m_centres.clone(), (Instance) obj.m_classCentre.copy());
	// obj.m_totalsize += newInst.size();
	//
	// // double oldDensity= this.m_classDensity[clusID];
	// // double mcSize= oldDensity*oldSize;
	// // this.m_classDensity[clusID]= updateDensity(clusID,
	// // newInst,mcSize);
	//
	// Instance oldSD = (Instance) obj.m_SD[pos].copy();
	// obj.m_SD[pos] = updateSD(newInst, oldSD, oldCentre, oldSize);
	// double SDDiff = CalculateED(oldSD, obj.m_SD[pos]);
	// double oldCapacity = m_sysCapacity;
	// this.m_sysCapacity += newInst.size();
	// double old_FarthestDis = obj.m_farthest_dis;
	// // obj.m_farthest_dis=
	// // updateFarthestDistance(obj.m_centres.clone());
	// obj.m_VDL = updateVDL(obj);
	// obj.m_maxVDL = getMaxLoss(obj.m_VDL);
	// double oldGF = obj.m_minGravF;
	// obj.m_minGravF = sysMinGF(classID);
	// double old_d_max[] = new double[this.d_max.length];
	// old_d_max = this.d_max.clone();
	// this.d_max = this.findMaxDistance();
	// s += " Centre Movement= " + centreMov + "\n" + " SD Diffrence= "
	// + SDDiff + "\n" + " OldBoundries= " + oldBoundry
	// + " New Boundries= " + obj.m_classBoundry[pos] + "\n"
	// + " Old Size= " + oldSize + " New Size= " + obj.m_size[pos]
	// + "\n" + " OldAverageDis= " + oldAvDist + " New Average= "
	// + obj.m_averageDistance[pos] + "\n" + " Old Capacity= "
	// + oldCapacity + " New Capacity= " + this.m_sysCapacity
	// + "\n";
	// // +" Old Farthest Distance= "+
	// // old_FarthestDis+" New= "+obj.m_farthest_dis+"\n"
	// // +"Old GF= "+ oldGF+" Updated GF= "+ obj.m_minGravF+"\n";
	// // s += " Centre Movement= " + centreMov + "\n";
	// // for (int n = 0; n < d_max.length; n++) {
	// // s += " D_max[" + n + "] = " + d_max[n] + " old: "
	// // + old_d_max[n] + "\n";
	// // }
	//
	// } else
	// s = " Failed to update";
	//
	// return s;
	//
	// }

	private double getMaxLoss(double[] loss) {
		double max = Double.MAX_VALUE;
		for (int i = 0; i < loss.length; i++)
			if (loss[i] < max)
				max = loss[i];
		return max;
	}

	public void setSlackSize(double size) {
		// if default size not set
		if (this.defaultSlackSize == -1)
			this.defaultSlackSize = size;
		this.slackSize = new double[this.m_noClasses];
		for (int i = 0; i < this.m_noClasses; i++)
			this.slackSize[i] = size;

	}

	public void setSlackSize(double[] size) {

		this.slackSize = new double[this.m_noClasses];
		this.slackSize = size.clone();
		if (this.defaultSlackSize == -1)
			this.defaultSlackSize = size[0];

	}

	public void setStableClusterSize(int size) {

		this.stableSize = size;

	}

	public void setBufferFlag(boolean b) {
		this.bufferSizeF = b;
	}

	public void setJPLFlag(boolean b) {
		this.JPLayerF = b;
	}

	public void setMovementThreshold(double threshold) {

		this.centreMovement = threshold;

	}

	public void setawaytemp(double at) {
		this.awaytemp = at;
	}

	private double[] updateVDL(ClassWSubClusters obj) {
		double[] loss = new double[obj.NoOfSubClusters];
		double orgDensity = densityFunction((double) obj.m_totalsize, obj.m_globalBoundry, 3);
		for (int j = 0; j < obj.NoOfSubClusters; j++) {
			int newsize = obj.m_totalsize - obj.m_size[j];
			double newRadius = getNewRadius(j, obj);
			double newDensity = densityFunction((double) newsize, newRadius, 3);
			loss[j] = newDensity - orgDensity;

		}
		return loss;
	}

	private double getNewRadius(int subClusterID, ClassWSubClusters obj) {

		Instance newCentre = getSubClustersCentre(subClusterID, obj);
		double boundry = 0;
		double temp = 0;
		for (int i = 0; i < obj.m_centres.length; i++) {
			if (i != subClusterID) {
				temp = CalculateED(newCentre, obj.m_centres[i]) + obj.m_classBoundry[i];
				if (temp > boundry)
					boundry = temp;
			}
		}
		return boundry;
	}

	private Instance getSubClustersCentre(int exCluster, ClassWSubClusters obj) {
		// Calculate Class centre with excluding sub-cluster

		Instance middle = (Instance) obj.m_centres[0].copy();
		double[] middleArray = new double[middle.numAttributes()];
		int n = obj.m_totalsize - obj.m_size[exCluster];

		for (int i = 0; i < middle.numAttributes(); i++) {
			for (int j = 0; j < obj.NoOfSubClusters; j++)
				if (j != exCluster)
					middleArray[i] += obj.m_centres[j].value(i) * obj.m_size[j];
			middleArray[i] /= (double) n;

		}
		for (int i = 0; i < middle.numAttributes(); i++) {
			middle.setValue(i, middleArray[i]);
		}
		return middle;
	}

	// private double updateFarthestDistance(Instance[] m_centres) {
	// double furthestDistance=-1;
	// double temp=0;
	// for(int i=0; i< m_centres.length;i++ )
	// {
	// for( int j=0; j< m_centres.length; j++){
	// if(i!=j){
	// temp= CalculateED(m_centres[i], m_centres[j]);
	// if( temp> furthestDistance)
	// furthestDistance= temp;
	// }
	// }
	// }
	//
	// return furthestDistance;
	// }

	private Instance updateGlobalCentre(Instance[] m_centres) {
		Instance middle = (Instance) m_centres[0].copy();
		double[] middleArray = new double[middle.numAttributes()];
		for (int i = 0; i < middle.numAttributes(); i++) {
			for (int j = 0; j < m_centres.length; j++)
				middleArray[i] += m_centres[j].value(i);
			middleArray[i] /= (double) m_centres.length;

		}
		for (int i = 0; i < middle.numAttributes(); i++) {
			middle.setValue(i, middleArray[i]);
		}
		return middle;

	}

	private double updateGlobalBoundry(double[] classBoundry, Instance[] centres, Instance classCentre) {
		double furthest = 0;
		double temp = 0;

		for (int i = 0; i < centres.length; i++) {
			temp = CalculateED(classCentre, centres[i]) + classBoundry[i];
			if (temp > furthest)
				furthest = temp;
		}
		return furthest;
	}

	private double updatedGlobalAvDistance(double[] m_averageDistance) {
		double Average = 0;

		for (int i = 0; i < m_averageDistance.length; i++) {
			Average = m_averageDistance[i];
		}
		Average /= m_averageDistance.length;

		return Average;
	}

	private double updateAvDistance(Instance newCentre, Instances newInst, double oldAvDist, double size) {
		double newAverage = 0;
		for (int i = 0; i < newInst.numInstances(); i++) {
			newAverage += CalculateED(newCentre, newInst.instance(i));
		}
		newAverage /= newInst.numInstances();
		// double w1= size/ ( size+ newInst.size() );
		// double w2= newInst.size()/ ( size+ newInst.size() );
		// double average=(w1* oldAvDist)+(w2*newAverage);
		double average = (oldAvDist + newAverage) / 2;

		return average;
	}

	private int chooseSubClusterToUpdate(ClassWSubClusters obj, Instances newInst) {

		return DisCand(obj, newInst);
	}

	private int vote(int c1, int c2, int c3, int c4) {
		int c = frequent(c1, c2, c3, c4);
		if (c != -1)
			return c;

		return c1;
	}

	private int frequent(int c1, int c2, int c3, int c4) {
		// TODO Auto-generated method stub
		if (c1 == c2 && c2 == c3)
			return c1;
		if (c1 == c3 && c3 == c4)
			return c1;
		if (c2 == c3 && c3 == c4)
			return c2;

		return -1;
	}

	private Instance getSD(Instances newInst) {

		Instance SDInstance = (Instance) newInst.firstInstance().copy();
		double SD[] = new double[newInst.numAttributes()];

		for (int i = 0; i < newInst.numAttributes(); i++)
			SD[i] = newInst.variance(i);

		for (int j = 0; j < SD.length; j++) {
			SD[j] = Math.sqrt(SD[j]);
			SDInstance.setValue(j, SD[j]);
		}

		return SDInstance;
	}

	private double getBoundry(Instances newInst) {

		if (newInst.numInstances() == 1)
			return 0;
		double maxDistance = 0;
		Instance cntrInst = (Instance) getInstancesCentre(newInst);
		double distance = 0;
		for (int i = 0; i < newInst.numInstances(); i++) {
			distance = CalculateED(cntrInst, newInst.instance(i));
			// distance= ED.distance(cntrInst, m_instances.get(i));

			if (distance > maxDistance)
				maxDistance = distance;
		}

		return maxDistance;

	}

	private int DisCand(ClassWSubClusters obj, Instances newInst) {

		double tmpDis = -1 * Double.MAX_VALUE;
		double dis = 0;
		int ID = -1;
		Instance centre = getInstancesCentre(newInst);
		for (int i = 0; i < obj.NoOfSubClusters; i++) {

			dis = CalculateED(obj.m_centres[i], centre);
			if (dis > tmpDis) {
				ID = i;
				tmpDis = dis;

			}
		}

		return ID;
	}

	// REVIEWED
	private Instance getInstancesCentre(Instances newInst) {
		// TODO Auto-generated method stub

		if (newInst.numInstances() == 1)
			return newInst.firstInstance();
		Instance InC = (Instance) newInst.firstInstance().copy();
		int m_numAtt = newInst.numAttributes();
		double[] mean = new double[m_numAtt];
		for (int i = 0; i < m_numAtt; i++) {
			mean[i] = newInst.meanOrMode(i);
		}
		for (int i = 0; i < m_numAtt; i++)
			InC.setValue(i, mean[i]);

		return InC;
	}

	// private double UpdateDensThreshold() {
	//
	// double densTh= m_classBoundry[0];
	// for( int i=1; i< m_noClasses;i++){
	// if(densTh< m_classBoundry[i])
	// densTh= m_classBoundry[i];
	// }
	// return densTh;
	// }

	// private double UpdateGravForce() {
	//
	//
	// double gravF=0, temp=0;
	// for( int n=0; n<this.m_noClasses;n++ ){
	// for( int i=n+1; i< this.m_noClasses;i++){
	// temp= gravetationForce(this.m_size[n], this.m_size[i], this.m_centres[n],
	// this.m_centres[i]);
	// if( temp>gravF)
	// gravF=temp;
	// }
	// }
	// return gravF;
	//
	// }

	private Instance updateSD(Instances newInst, Instance prevSD, Instance oldCentre, int oldSize) {

		Instance updatedMean = (Instance) oldCentre.copy();
		int n = oldSize;
		Instance nIns = (Instance) newInst.firstInstance().copy();
		double[] mean = new double[nIns.numAttributes()];
		double[] M2 = new double[nIns.numAttributes()];
		Instance updatedSD = (Instance) prevSD.copy();
		for (int i = 0; i < M2.length; i++) {
			M2[i] = prevSD.value(i) * prevSD.value(i) * (n);
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
			M2[i] = M2[i] / (n - 1);

			updatedSD.setValue(i, Math.sqrt(M2[i]));
		}

		return updatedSD;

	}

	private double updateBoundry(Instance m_centres, Instances newInst, double oldBoundry) {
		// TODO Auto-generated method stub
		Instances m_instances = new Instances(removeClass(newInst));
		double maxDistance = oldBoundry;
		Instance newCntr = (Instance) m_centres.copy();
		double distance = 0;
		for (int i = 0; i < m_instances.numInstances(); i++) {

			distance = CalculateED(newCntr, m_instances.instance(i));
			if (distance > maxDistance)
				maxDistance = distance;
		}
		return maxDistance;
	}

	private Instance updateCentre(int size, Instances newInst, Instance oldCentre) {
		// TODO Auto-generated method stub
		Instance updatedMean = (Instance) oldCentre.copy();
		double n = size;
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

	String S_Novelty;
	private boolean suspectedDrift = false;
	public ArrayList<Integer> driftClassID = new ArrayList<Integer>();
	public int stableSize = -1;
	public double centreMovement = 0;
	public boolean bufferSizeF = true;
	public boolean JPLayerF = true;
	public double slackSize[];
	protected double defaultSlackSize = -1;
	boolean updFlag = false;
	boolean validFlag = false;
	private Instances dataWarehouse;
	public String ALStats = "";
	public boolean faNovelFlag = false;
	public boolean reNovelFlag = false;
	private String orgLabel = "";
	private Instances justPredicted;

	public NovelPredection noveltyStatistics() {
		// Novel label is true if the label is not exist in the Base Model.
		// False if it is already exist
		// b is the prediction flag
		S_Novelty = "";
		boolean novelLabel = true;
		boolean unknown = false;
		Instances newData = removeClass(this.dataWarehouse);
		boolean recurrent = this.reNovelFlag;
		boolean fa = this.faNovelFlag;
		// boolean densityBoolean=noveltyDetection_density(newData,
		// this.CWSCModelOld);
		// boolean gravBoolean= noveltyDetection_gravity(newData,
		// this.CWSCModelOld);

		String[] newLabel2 = getMajorityLabel(this.dataWarehouse);
		String newLabel = newLabel2[0];

		// Tp for already new born novel class

		// if( novelFlag && newLabel== this.orgLabel) {
		// NovelPredection NP= new NovelPredection(novelFlag, S_Novelty);
		// novelInstancesCounter(NP);
		// NP.setDenNovFlag(densityBoolean);
		// NP.setGravNovFlag(gravBoolean);
		// NP.setNoveltyType("Tp");
		// NP.setClassSize(newData.size());
		// NP.setClassTrueLabel(newLabel2);
		// return NP;
		// }

		for (int j = 0; j < this.m_Oldlabels.length; j++)
			if (this.m_Oldlabels[j].trim().equalsIgnoreCase(newLabel.trim()))
				novelLabel = false;

		if (m_predLabel.trim().equalsIgnoreCase("unknown")) {
			unknown = true;
			novelLabel = false;
		}
		// Distance>> Density>> Gravity>> SD

		boolean voted = voteForNovelty(fa, true, true, true);

		boolean[] Measures = { voted, true, true, true };
		String[] S = new String[4];

		for (int j = 0; j < Measures.length; j++) {
			if (novelLabel && Measures[j])
				S[j] = "Tp";
			else if (novelLabel)
				S[j] = "Fn";
			else if (Measures[j])
				S[j] = "Fp";
			else
				S[j] = "Tn";
		}

		// Set Novelty detection statistics
		// Predicted
		// Novel Existing
		// Novel Tp Fn
		// True
		// Existing Fp Tn
		//
		//
		NovelPredection NP = new NovelPredection(fa, recurrent, "");
		NP.setm_measNoveltyType(S);
		novelInstancesCounter(NP);
		NP.setDenNovFlag(true);
		NP.setGravNovFlag(true);
		if (unknown)
			NP.setUnknown();
		else if (novelLabel && (fa || recurrent)) {

			NP.setNoveltyType("Tp");
		} else if (novelLabel)
			NP.setNoveltyType("Fn");
		else if (fa || recurrent)
			NP.setNoveltyType("Fp");
		else
			NP.setNoveltyType("Tn");

		NP.setPredictedLabel(this.m_predLabel);

		NP.setClassSize(newData.numInstances());

		NP.setClassTrueLabel(newLabel2);

		return NP;

	}

	private void novelInstancesCounter(NovelPredection nP) {

		Instances ins = new Instances(this.dataWarehouse);
		ins.setClassIndex(ins.numAttributes() - 1);
		String[] clusterLabels = new String[ins.numClasses()];
		int f = 0;
		for (Enumeration c = ins.classAttribute().enumerateValues(); c.hasMoreElements();) {

			clusterLabels[f] = ((String) c.nextElement()).trim();
			f++;
		}

		int novel = 0;
		boolean exist = false;
		for (int i = 0; i < this.dataWarehouse.numInstances(); i++) {
			for (int j = 0; j < this.m_Oldlabels.length; j++)
				if (this.m_Oldlabels[j].trim()
						.equalsIgnoreCase(clusterLabels[(int) dataWarehouse.instance(i).classValue()].trim()))
					exist = true;
			if (!exist)
				novel++;
		}
		nP.setNoofNovelInstances(novel);

	}

	private String[] getMajorityLabel(Instances ins) {

		// TODO Auto-generated method stub
		ins.setClassIndex(ins.numAttributes() - 1);
		int[] counter = new int[ins.numClasses()];
		String[] clusterLabels = new String[ins.numClasses()];
		int f = 0;
		for (Enumeration c = ins.classAttribute().enumerateValues(); c.hasMoreElements();) {

			clusterLabels[f] = ((String) c.nextElement()).trim();
			f++;
		}

		for (int i = 0; i < ins.numInstances(); i++) {
			double s = ins.instance(i).classValue();
			boolean n;
			if ((int) s == 3)
				n = false;
			if (s != -1) {

				counter[(int) s]++;
			}
		}
		int max = 0;
		int s = -1;
		for (int i = 0; i < counter.length; i++) {
			while (counter[i] > max) {
				s = i;
				max = counter[i];
			}
		}
		String[] clusterLabel = new String[2];
		clusterLabel[0] = clusterLabels[s];
		clusterLabel[1] = Integer.toString(max);

		return clusterLabel;

	}

	private boolean voteForNovelty(boolean b1, boolean b2, boolean b3, boolean b4) {
		S_Novelty += " Distance Vote: " + b1 + "|| Density Vote: " + b2 + "||Gravity Vote: " + b3 + "|| SD Vote: " + b4
				+ "\n";
		if (majortyTrue(b1, b2, b3, b4)) {
			S_Novelty += " Novel Class detected due to the majority vote" + "\n";
			return true;
		}
		S_Novelty += " Failed to detected any novel class" + "\n";
		return false;
	}

	private boolean majortyTrue(boolean b1, boolean b2, boolean b3, boolean b4) {

		// gives priority to distance measure if 2 are equls
		int counter = 0;
		if (b1)
			counter++;
		if (b2)
			counter++;
		if (b3)
			counter++;
		// if(b4) counter++;
		//
		if (counter == 3 || counter == 2)
			return true;
		// if( counter>=3) return true;
		return false;
	}

	private double sysMinGF(int i) {

		double temp = 0;
		double GF = Double.MAX_VALUE;
		for (int j = 0; j < this.m_classesWithClusters.get(i).NoOfSubClusters; j++) {
			for (int k = 0; k < this.m_classesWithClusters.get(i).NoOfSubClusters; k++) {
				if (k != j) {
					temp = gravetationForce(this.m_classesWithClusters.get(i).m_size[k],
							this.m_classesWithClusters.get(i).m_size[j],
							(Instance) this.m_classesWithClusters.get(i).m_centres[k].copy(),
							(Instance) this.m_classesWithClusters.get(i).m_centres[j].copy());
					if (temp < GF)
						GF = temp;
				}
			}

		}

		return GF;
	}

	public String[] getLabels() {
		return this.m_labels;
	}

	public String[] getOldLabels() {
		return this.m_Oldlabels;
	}

	public int getNumberOfClasses() {
		return m_noClasses;
	}

	public ArrayList<ClassWSubClusters> getClassWithSubClustersArray() {

		return m_classesWithClusters;
	}

	boolean reset = false;
	boolean accumlate = false;
	private int cutoff_index = -1;
	public double awaytemp = 0.1;

	public String adaptationComponent(Instances nInst) throws Throwable {

		this.m_predLabel = "";
		// Change to get the time stamp
		this.reNovelFlag = false;
		this.faNovelFlag = false;
		boolean recent = true;
		Instances newInst = new Instances(nInst);
		String str = "";
		String details = "";

		if (bufferSizeF) {

			if (this.suspNovelInstances.numInstances() > this.stableSize * 10) {
				details += "Case 0.0, Buffer Size Case,";
				casesCounter[0][0]++;
				if (accumlate)
					dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
				else
					this.dataWarehouse = new Instances(this.suspNovelInstances);

				accumlate = false;
				this.cutoff_index = -1;
				int index = declareUnCorelatedInstances(this.suspNovelInstances);
				if (index == -1) {
					m_predLabel = "Unknown";
					details += " Previous Unknown ,";
				} else if (this.m_classesWithClusters.get(index).m_classID == -1) {
					this.accumalateNBClass(this.suspNovelInstances, index);
					details += ", New born class accumalated ,";
					reNovelFlag = true;
				} else
					m_predLabel = m_Oldlabels[index];

				this.justPredicted = new Instances(this.dataWarehouse, 0);
				this.suspectedDrift = false;
				this.driftClassID.clear();
				this.suspNovelInstances.delete();
				str += details;
				return str;

			}
		}

		if (this.JPLayerF) {
			if (this.suspNovelInstances.numInstances() == 0) {
				// Check if empty
				boolean coF = instancesCorrelated2(newInst, justPredicted);
				if (this.justPredicted.numInstances() == 0 || (this.justPredicted.numInstances() != 0 && !coF)) {
					if (reset) {
						this.suspectedDrift = false;
						this.driftClassID.clear();
					}
					accumlate = false;
					cutoff_index = -1;
					details += " Case 0.2, ";
					casesCounter[0][2]++;
					str += details;
					this.justPredicted = new Instances(newInst, 0);
					str += processNewDatawithNoHistory(newInst);
					return str;
				}

				if (coF && this.justPredicted.numInstances() != 0) {
					accumlate = true;
					cutoff_index = justPredicted.numInstances();
					this.suspNovelInstances = new Instances(justPredicted);
					details += " Case 0.3: seeding with  t-1 prediction, ";
					casesCounter[0][3]++;
					this.suspectedDrift = false;
					this.driftClassID.clear();
					this.justPredicted = new Instances(suspNovelInstances, 0);
					str += details;
				}

			} else if (this.suspNovelInstances.numInstances() != 0 && this.justPredicted.numInstances() == 0) {

				Instances accInst;
				if (accumlate) {
					accInst = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
				} else
					accInst = new Instances(this.suspNovelInstances);

				if (!instancesCorrelated2(newInst, accInst))

				{
					details += "Case 0.1, New data is not correlated to accumlated,";
					casesCounter[0][1]++;
					accumlate = false;
					this.cutoff_index = -1;
					int index = declareUnCorelatedInstances(accInst);
					if (index == -1) {
						m_predLabel = "Unknown";
						details += " Previous Unknown ,";
					} else if (this.m_classesWithClusters.get(index).m_classID == -1) {
						this.accumalateNBClass(accInst, index);
						details += ", New born class accumalated ,";
						reNovelFlag = true;
					} else
						m_predLabel = m_Oldlabels[index];
					this.dataWarehouse = new Instances(accInst);
					this.justPredicted = new Instances(accInst, 0);
					this.suspectedDrift = false;
					this.driftClassID.clear();
					this.suspNovelInstances.delete();
					str += details;

					return str;

				}

				else {
					details += "Case 0.4, New data is  correlated to accumlated, Keep going ";
					this.justPredicted = new Instances(accInst, 0);
					str += details;
					casesCounter[0][4]++;
				}

			} else if (this.suspNovelInstances.numInstances() != 0 && this.justPredicted.numInstances() != 0) {

				Instances accInst;
				if (accumlate)
					accInst = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);

				else
					accInst = new Instances(this.suspNovelInstances);

				if (!instancesCorrelated2(newInst, accInst) && instancesCorrelated2(newInst, justPredicted)) {
					details += "Case 0.5, New data  correlated with JP, incorr with buffer,";
					casesCounter[0][5]++;
					accumlate = false;
					this.cutoff_index = -1;
					int index = declareUnCorelatedInstances(accInst);
					if (index == -1) {
						m_predLabel = "Unknown";
						details += " Previous Unknown ,";
					} else if (this.m_classesWithClusters.get(index).m_classID == -1) {
						this.accumalateNBClass(accInst, index);
						details += ", New born class accumalated ,";
						reNovelFlag = true;
					} else
						m_predLabel = m_Oldlabels[index];
					this.dataWarehouse = new Instances(accInst);
					this.suspectedDrift = false;
					this.driftClassID.clear();
					accumlate = true;
					cutoff_index = justPredicted.numInstances();
					this.suspNovelInstances = new Instances(justPredicted);
					this.justPredicted = new Instances(accInst, 0);
					str += details;
					return str;

				}
			}
		}
		// Old one... With no JP layer
		else {
			if (this.suspNovelInstances.numInstances() == 0) {
				// Chcek if empty
				boolean coF = instancesCorrelated2(newInst, justPredicted);
				if (this.justPredicted.numInstances() == 0 || (this.justPredicted.numInstances() != 0 && !coF)) {
					if (reset) {
						this.suspectedDrift = false;
						this.driftClassID.clear();
					}
					accumlate = false;
					cutoff_index = -1;
					details += " Case 0.2, ";
					casesCounter[0][2]++;
					str += details;
					this.justPredicted = new Instances(newInst, 0);
					str += processNewDatawithNoHistory(newInst);
					return str;
				}

				if (coF && this.justPredicted.numInstances() != 0) {
					accumlate = true;
					cutoff_index = justPredicted.numInstances();
					this.suspNovelInstances = new Instances(justPredicted);
					details += " Case 0.3: seeding with  t-1 prediction, ";
					casesCounter[0][3]++;
					this.suspectedDrift = false;
					this.driftClassID.clear();
					this.justPredicted = new Instances(suspNovelInstances, 0);
					str += details;
				}

			}
		}
		if (!accumlate && reset) {
			this.suspectedDrift = false;
			this.driftClassID.clear();
		}
		details = "";
		// Case2, Case 3:
		// If the buffer is not empty- Susp Novel or concept drift exist in the
		// buffer
		String strDis = "";

		Instance NovelCentre = getInstancesCentre(removeClass(newInst));
		String[] MLabel = getMajorityLabel(newInst);
		str += MLabel[1] + "||" + newInst.numInstances() + "," + MLabel[0] + ",";
		boolean[] inside = new boolean[this.m_noClasses];
		boolean[] inSlack = new boolean[this.m_noClasses];
		String insideStr = "";
		String inSlackStr = "";

		// Compare the distance from new Instances to the class centre with the
		// max distance inside the class
		// ERROR>>> When the concept drift is realted to other class ( buffer
		// concept drift is diffrent from new data concept drift)
		double[] SZ = new double[this.m_noClasses];

		for (int i = 0; i < this.m_noClasses; i++) {

			inside[i] = false;
			inSlack[i] = false;
			// if(
			// this.m_classesWithClusters.get(i).getLable().toLowerCase().contains("walk"))
			// SZ[i]= 900;
			// else
			// if(
			// this.m_classesWithClusters.get(i).getLable().toLowerCase().contains("sit"))
			// SZ[i]= 3500;
			// else
			SZ[i] = this.slackSize[i];

			double distanceDiff = CalculateED(NovelCentre, this.m_classesWithClusters.get(i).m_classCentre);
			strDis += SZ[i] + "," + distanceDiff + ",";
			if (distanceDiff < this.d_max[i])
				inside[i] = true;
			else if (distanceDiff - this.d_max[i] < SZ[i])
				inSlack[i] = true;
			insideStr += inside[i] + ",";
			inSlackStr += inSlack[i] + ",";

		}
		int hitInsideID = IDofInsideHit(inside, NovelCentre);
		int inSlackID = IDofInsideHit(inSlack, NovelCentre);
		str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
		for (int id = 0; id < driftClassID.size(); id++)
			str += "DID= " + this.driftClassID.get(id) + ",";
		str += insideStr + inSlackStr;
		str += hitInsideID + "," + inSlackID + ",";
		str += strDis;

		// Inside any?
		// Case 2: If the buffer is not empty with recent data & New data is
		// Inside
		// Declared as concept drift and empty the buffer

		if (hitInsideID != -1 && recent) {

			// Case 2.1:
			// "Inside hit" for concept drift. When prev. data in buffer is
			// suspeced concept drift and there is hit inside following points
			if (this.suspectedDrift && this.driftClassID.contains(hitInsideID)) {
				details += "Case 2.1, New data and data in the buffer are classified as concept drift as' Inside hit',";
				casesCounter[2][1]++;
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				if (accumlate) {
					dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
					accumlate = false;
					cutoff_index = -1;
				} else
					dataWarehouse = new Instances(this.suspNovelInstances);

				this.justPredicted = new Instances(this.dataWarehouse);
				if (this.m_classesWithClusters.get(hitInsideID).m_classID == -1) {
					this.accumalateNBClass(this.suspNovelInstances, hitInsideID);
					details += ", New born class accumalated ,";
					reNovelFlag = true;
				} else
					m_predLabel = m_Oldlabels[hitInsideID];
				reset = true;
				this.suspNovelInstances.delete();
				str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
				for (int id = 0; id < driftClassID.size(); id++)
					str += "DID= " + this.driftClassID.get(id) + ",";
				str += details + "\n";
				return str;
			}
			// Case 2.2 : When there is no suspected drift flag, However, the
			// accumlated data( new and buffer) is inside the cluster.
			Instance Inst = mergeCentre(removeClass(newInst), removeClass(this.suspNovelInstances));
			double distance = CalculateED(Inst, this.m_classesWithClusters.get(hitInsideID).m_classCentre);
			// Merged centre in Slack
			if (distance - this.d_max[hitInsideID] < SZ[hitInsideID]) {
				details += "Case 2.2,  Distance between merged centre and  class" + hitInsideID + " is: " + distance
						+ ",";
				casesCounter[2][2]++;
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				if (accumlate) {
					dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
					accumlate = false;
					cutoff_index = -1;
				} else
					dataWarehouse = new Instances(this.suspNovelInstances);

				this.justPredicted = new Instances(this.dataWarehouse);
				if (this.m_classesWithClusters.get(hitInsideID).m_classID == -1) {
					this.accumalateNBClass(this.suspNovelInstances, hitInsideID);
					details += ", New born class accumalated";
					this.reNovelFlag = true;
				} else
					m_predLabel = m_Oldlabels[hitInsideID];

				str += " Drift info" + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ","
						+ this.driftClassID;
				for (int id = 0; id < driftClassID.size(); id++)
					str += "DID= " + this.driftClassID.get(id) + ",";
				reset = true;
				this.suspNovelInstances.delete();
				str += details + "\n";
				return str;
			}

			// Case 2.3: when there is no relation between new data and data in
			// buffer
			details += "Case 2.3, New data is recognised as inside class " + hitInsideID
					+ " with no relation to the data in buffer" + ",";
			casesCounter[2][3]++;
			this.dataWarehouse = new Instances(newInst);

			if (this.suspNovelInstances.numInstances() == this.cutoff_index) {
				this.justPredicted = new Instances(dataWarehouse);
				this.suspNovelInstances.delete();
				reset = true;
			}

			this.reNovelFlag = false;
			if (this.m_classesWithClusters.get(hitInsideID).m_classID == -1) {
				this.accumalateNBClass(newInst, hitInsideID);
				details += ", New born class accumalated";
				reNovelFlag = true;
			} else
				m_predLabel = m_Oldlabels[hitInsideID];

			str += " Drift info:" + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
			for (int id = 0; id < driftClassID.size(); id++)
				str += "DID= " + this.driftClassID.get(id) + ",";
			str += details + "\n";
			return str;
		}
		if (inAnySlack(inSlack)) {
			// Cases when new data is inside the slack of one of the classes

			// Case 3.1: New: Inslack, Old: inSlack

			boolean duplicated = false;
			boolean inSlackf = false;
			int y = -1;
			int m = 0;
			int d = 0;
			duplicated = false;
			// Special case: Overlapped slakes classes
			for (d = 0; d < this.m_noClasses; d++) {
				if (duplicated)
					break;
				m = 0;
				if (inSlack[d] && recent && this.driftClassID.contains(d) && this.suspectedDrift) {
					inSlackf = true;
					y = d;
					for (m = d + 1; m < this.m_noClasses; m++) {
						if (inSlack[m] && recent && this.driftClassID.contains(m) && this.suspectedDrift) {
							duplicated = true;
							break;
						}
					}
				}
			}

			if (duplicated) {
				this.suspectedDrift = true;
				if (!this.driftClassID.contains(y))
					this.driftClassID.add(y);
				if (!this.driftClassID.contains(m))
					this.driftClassID.add(m);
				details += "Case 3.0 : Duplicated case";
				casesCounter[3][0]++;
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				this.justPredicted = new Instances(newInst, 0);
				str += " Drift info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
				for (int id = 0; id < driftClassID.size(); id++)
					str += "DID= " + this.driftClassID.get(id) + ",";
				str += details + "\n";
				details = "";
				return str;

				// y= chooseOverlapClasses(NovelCentre,y,m);
			}
			// Case 3.1:
			if (inSlackf) {
				details += "Case 3.1, New data and data in the buffer are classified as a drift,";
				casesCounter[3][1]++;
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				if (accumlate) {
					dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
					accumlate = false;
					cutoff_index = -1;
				}

				else
					dataWarehouse = new Instances(this.suspNovelInstances);
				this.justPredicted = new Instances(dataWarehouse);
				if (this.m_classesWithClusters.get(y).m_classID == -1) {
					this.accumalateNBClass(this.suspNovelInstances, y);
					details += "New born class accumalated ,";
					reNovelFlag = true;
				} else {
					m_predLabel = m_Oldlabels[y];
				}
				reset = true;
				this.suspNovelInstances.delete();
				this.suspectedDrift = false;
				this.driftClassID.clear();
				str += " Drift info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
				str += details + "\n";
				return str;
			}

			boolean case3_2 = false;
			String info = "";
			// Case 3.2: New: Inslack, Old: outside & Merge: Inslack
			// DISPUTE CASE: Put as suspected or declared drift?
			int ID = -1;
			for (int i = 0; i < this.m_noClasses; i++) {
				Instance Inst = mergeCentre(removeClass(newInst), removeClass(this.suspNovelInstances));
				double distance = CalculateED(Inst, this.m_classesWithClusters.get(i).m_classCentre);
				// Merged centre in Slack
				if (inSlack[i] && distance - this.d_max[i] < SZ[i]) {
					case3_2 = true;
					this.suspectedDrift = true;
					if (!this.driftClassID.contains(i))
						this.driftClassID.add(i);
					info += "  MC Distance to " + i + " = " + distance + ",";
					ID = i;
					break;
				}
			}
			if (case3_2) {
				details += "Case 3.2" + info;
				casesCounter[3][2]++;
				// this.suspNovelInstances.addAll(newInst);
				// this.justPredicted = new Instances(newInst, 0);
				// str += " Drift info: " + this.suspNovelInstances.size() + ","
				// + this.suspectedDrift + ",";
				// for (int id = 0; id < driftClassID.size(); id++)
				// str += "DID= " + this.driftClassID.get(id) + ",";
				// str += details + "\n";
				// details = "";
				// return str;
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				if (accumlate) {
					dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
							this.suspNovelInstances.numInstances() - cutoff_index);
					accumlate = false;
					cutoff_index = -1;
				} else
					dataWarehouse = new Instances(this.suspNovelInstances);
				this.justPredicted = new Instances(dataWarehouse);
				if (this.m_classesWithClusters.get(ID).m_classID == -1) {
					this.accumalateNBClass(this.suspNovelInstances, ID);
					details += ", New born class accumalated ,";
					reNovelFlag = true;
				} else
					m_predLabel = m_Oldlabels[ID];

				this.suspNovelInstances.delete();
				str += this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
				str += details + "\n";
				return str;

				// this.suspectedDrift=true;
				// //
				// this.justPredicted= new Instances(newInst,0);
				// str+=" Drift info:
				// "+this.suspNovelInstances.size()+","+this.suspectedDrift+",";
				// for( int id=0; id< driftClassID.size();id++)
				// str+="DID= "+ this.driftClassID.get(id)+",";
				// str+=details+"\n";
				// return str;

			}

			// Case 3.3: New: Inslack, Old: outside
			if (inAnySlack(inSlack)) {
				if (instancesCorrelated2(this.suspNovelInstances, newInst)) {
					details += "Case 3.3,  Correlated..then merge and keep";
					casesCounter[3][3]++;
					this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
					this.suspectedDrift = true;
					for (int i = 0; i < this.m_noClasses; i++) {
						if (inSlack[i]) {
							if (!this.driftClassID.contains(i)) {
								this.driftClassID.add(i);
							}

						}

					}
					this.justPredicted = new Instances(newInst, 0);
					str += " Drift info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
					for (int id = 0; id < driftClassID.size(); id++)
						str += "DID= " + this.driftClassID.get(id) + ",";
					str += details + "\n";
					return str;
				}
				String newStr = declareUnkownInstance(this.suspNovelInstances);
				details += "Case 3.4,  not Correlated..then declare old and reinitiate with newData" + newStr;
				casesCounter[3][4]++;
				this.dataWarehouse = new Instances(this.suspNovelInstances);
				this.justPredicted = new Instances(this.suspNovelInstances, 0);
				if (newStr.contains("novel"))
					this.faNovelFlag = true;
				else
					this.faNovelFlag = false;

				this.suspNovelInstances.delete();
				this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
				this.suspectedDrift = true;
				this.driftClassID.clear();
				this.m_predLabel = "Unknown";
				for (int i = 0; i < this.m_noClasses; i++) {
					if (inSlack[i]) {
						if (!this.driftClassID.contains(i)) {
							this.driftClassID.add(i);
						}
					}
				}
				str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
				for (int id = 0; id < driftClassID.size(); id++)
					str += "DID= " + this.driftClassID.get(id) + ",";
				str += details + "\n";
				return str;

			}
		}

		String novelInfo = "";
		String predetails = details;

		// We reach this point in the code if the buffer is not empty and the
		// new data is suspected novel
		// New data is outside the existing classes and outside the slack too

		// Case 4:1: New: Outside, Old: Inslack
		// If merged centre is in slack
		details = "";
		Instance mergedCentre = mergeCentre(removeClass(newInst), removeClass(this.suspNovelInstances));
		boolean case4_1 = false;
		String info = "";
		int ID = -100;
		if (this.suspectedDrift) {
			for (int i = 0; i < this.m_noClasses; i++) {

				if (this.driftClassID.contains(i)) {

					double distance = CalculateED(mergedCentre, this.m_classesWithClusters.get(i).m_classCentre);
					if (distance - this.d_max[i] < SZ[i]) {
						case4_1 = true;
						this.suspectedDrift = true;
						if (!this.driftClassID.contains(i))
							this.driftClassID.add(i);
						info += "  MC Distance to " + i + " = " + distance + ",";
						ID = i;
						break;
					}
				}
			}
		}

		if (case4_1) {
			details += "Case 4.1" + info;
			casesCounter[4][1]++;
			// this.suspNovelInstances.addAll(newInst);
			// this.justPredicted = new Instances(newInst, 0);
			// str += " Drift Info : " + this.suspNovelInstances.size() + ","
			// + this.suspectedDrift + ",";
			// for (int id = 0; id < driftClassID.size(); id++)
			// str += "DID= " + this.driftClassID.get(id) + ",";
			// str += details + "\n";
			// return str;

			// Modified to be the same as case 3.2
			this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
			if (accumlate) {
				dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
						this.suspNovelInstances.numInstances() - cutoff_index);
				accumlate = false;
				cutoff_index = -1;
			} else
				dataWarehouse = new Instances(this.suspNovelInstances);
			this.justPredicted = new Instances(dataWarehouse);
			if (this.m_classesWithClusters.get(ID).m_classID == -1) {
				this.accumalateNBClass(this.suspNovelInstances, ID);
				details += ", New born class accumalated ,";
				reNovelFlag = true;
			} else
				m_predLabel = m_Oldlabels[ID];
			reset = true;
			this.suspNovelInstances.delete();

			str += this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
			str += details + "\n";
			return str;
		}

		// Case 4.2 and 4.3:
		boolean away[] = new boolean[this.m_noClasses];
		boolean c1 = false, c2 = false, c3 = false;

		// moving away ( newDistance >oldDistance) Or the centre movement
		// threshold)
		// Both cases result a new clusetr
		Instance previousCentre = (Instance) getInstancesCentre(removeClass(this.suspNovelInstances)).copy();
		;
		// if(accumlate)
		// previousCentre= (Instance) getInstancesCentre(new
		// Instances(this.suspNovelInstances, cutoff_index,
		// this.suspNovelInstances.size()-cutoff_index));

		// Instances tempInsts= new Instances(this.suspNovelInstances);
		// tempInsts.addAll(newInst);
		// Instance mergedNovelCentre=(Instance)
		// getInstancesCentre(removeClass(tempInsts));
		this.oldNoveldensity = OldDensityfn(removeClass(this.suspNovelInstances), previousCentre);

		for (int i = 0; i < this.m_noClasses; i++) {
			away[i] = false;
			// There is 2 ways to calculate the new centre. Either with new data
			// only or with the accumalated data
			// The one used here is for new data only

			double newDistance = CalculateED(NovelCentre, this.m_classesWithClusters.get(i).m_classCentre);
			double oldDistance = CalculateED(previousCentre, this.m_classesWithClusters.get(i).m_classCentre);
			double temp = newDistance - oldDistance;
			novelInfo += temp + " ,";
			// Moving away or a little bit closer( almost the same)
			if (temp > -(this.awaytemp)) {
				away[i] = true;

				for (int j = 0; j < this.driftClassID.size(); j++) {
					if (driftClassID.get(j) == i)
						driftClassID.remove(j);
				}
				if (driftClassID.size() == 0)
					this.suspectedDrift = false;

			}
		}

		boolean obsNovel = true;
		for (int i = 0; i < this.m_noClasses; i++) {
			if (away[i] == false)
				obsNovel = false;
		}
		// Adding to the buffer,
		// The buffer will be cleared if the stability condition satisfied
		this.suspNovelInstances = merge(this.suspNovelInstances, newInst);

		// If it is away from all, then it is added to suspecious novel,
		// If stable, it will be declared as novel
		if (obsNovel) {
			this.novelClassCentre = accumalateCentre(NovelCentre, removeClass(newInst));

			// str+="\n Moving away from all, egment with label "+ MLabel[0]+
			// " with size "+ newInst.size()+" and majority "+
			// MLabel[1]+" is suspected to be novel";

			// Check stability

			if (this.suspNovelInstances.numInstances() > this.stableSize)
				c1 = true;
			novelInfo += c1 + ",";
			// centre movement
			double temp = 0;
			temp = CalculateED(previousCentre, this.novelClassCentre);

			if (temp < centreMovement)
				c2 = true;
			novelInfo += temp + "," + c2 + ",";
			// Density
			// str+="\n suspNovelInstances size= "+ suspNovelInstances.size();
			double density = OldDensityfn(removeClass(this.suspNovelInstances), this.novelClassCentre);

			if ((density / this.oldNoveldensity) > 0.9)
				c3 = true;
			novelInfo += this.oldNoveldensity + "," + density + "," + c3 + ",";
			this.oldNoveldensity = density;

			if (c1 && c2 && c3) {
				details += " Stable class satisied all conditions ,";

				try {
					if (accumlate) {
						dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
								this.suspNovelInstances.numInstances() - cutoff_index);
						accumlate = false;
						cutoff_index = -1;
					} else
						this.dataWarehouse = new Instances(this.suspNovelInstances);
					this.justPredicted = new Instances(this.dataWarehouse);
					this.faNovelFlag = true;
					this.addNovelClass(suspNovelInstances, MLabel[0]);
					details += "Case 4.3, Novel Class created ,";
					details += ALStats;
					casesCounter[4][3]++;
					str += novelInfo;
					this.suspNovelInstances.delete();

					reset = true;
					str += details + "\n";
					return str;

				} catch (Exception e) {

					e.printStackTrace();
				}
			}
		} else {
			boolean closer = false;
			String classes = " ";
			for (int i = 0; i < this.m_noClasses; i++) {
				if (away[i] == false) {
					closer = true;

					if (!this.driftClassID.contains(new Integer(i))) {
						this.driftClassID.add(i);
						classes += i + "A ";
					}
				}
			}

			if (closer) {
				this.justPredicted = new Instances(newInst, 0);
				this.suspectedDrift = true;
				details += ", Case 4.2, Getting closer to classes" + classes;
				casesCounter[4][2]++;
				str += novelInfo;
				str += details + "\n";
				return str;
			}
		}

		str += predetails;
		str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
		for (int id = 0; id < driftClassID.size(); id++)
			str += "DID= " + this.driftClassID.get(id) + ",";
		this.justPredicted = new Instances(newInst, 0);
		str += "Case 4.4, suspected novel class is not stable yet! ,";
		casesCounter[4][4]++;
		str += novelInfo;
		str += details + "\n";
		return str;

	}

	private int chooseOverlapClasses(Instance novelCentre, int i, int j) {
		// TODO Auto-generated method stub
		double distance1 = CalculateED(novelCentre, this.m_classesWithClusters.get(i).m_classCentre);
		double distance2 = CalculateED(novelCentre, this.m_classesWithClusters.get(j).m_classCentre);
		if (distance1 < distance2)
			return i;
		return j;
	}

	private int declareUnCorelatedInstances(Instances novel_instances) {

		if (this.driftClassID.size() == 1) {
			return this.driftClassID.get(0);
		}
		double distance = Double.MAX_VALUE;
		int ID = -100;
		if (this.driftClassID.size() > 1) {
			// Choose closeset

			for (int i = 0; i < this.driftClassID.size(); i++) {
				double temp = CalculateED(getInstancesCentre(removeClass(novel_instances)),
						this.m_classesWithClusters.get(driftClassID.get(i)).m_classCentre);
				if (temp < distance) {
					distance = temp;
					ID = driftClassID.get(i);
				}

			}
			return ID;
		}
		// Suspected novel or unknown
		return -1;
	}

	private int IDofInsideHit(boolean[] inside, Instance novelCentre) {

		int index = -1;
		int counter = 0;
		for (int i = 0; i < inside.length; i++) {
			if (inside[i]) {
				counter++;
				index = i;
			}
		}
		// If it is inside only one class
		if (counter == 1)
			return index;

		// If it is inside multiple, choose the closest one

		double temp = Double.MAX_VALUE;
		for (int i = 0; i < this.m_noClasses; i++) {
			double distanceDiff = CalculateED(novelCentre, this.m_classesWithClusters.get(i).m_classCentre);
			// double farInside= this.d_max[i]- distanceDiff;
			if (inside[i] && distanceDiff < temp) {
				temp = distanceDiff;
				index = i;
			}
		}
		return index;
	}

	private Instance mergeCentre(Instances newInst, Instances allInst) throws Exception {

		allInst = merge(allInst, newInst);
		return this.getInstancesCentre(allInst);
	}

	private boolean inAnySlack(boolean[] inSlack) {
		// TODO Auto-generated method stub
		for (int i = 0; i < inSlack.length; i++) {
			if (inSlack[i])
				return true;
		}
		return false;
	}

	private String declareUnkownInstance(Instances inst) {
		// TODO Auto-generated method stub
		String str = "";
		if (inst.numInstances() > this.stableSize) {
			return str = ", Prev. data is novel with uncertainity, ";
		}
		return str += ", Prev. data Unknown  with uncertainity,";

	}

	// public double[][] getCorrelationCoefficients() {
	// if (this.correlationCoefficients == null) {
	// this.numAttributes = this.instances.numAttributes();
	// this.correlationCoefficients = new double[numAttributes][numAttributes];
	// for (int i = 0 ; i < numAttributes ; i++) {
	// double[] event1 = instances.attributeToDoubleArray(i);
	// for (int j = 0 ; j < i ; j++) {
	// double[] event2 = instances.attributeToDoubleArray(j);
	// this.correlationCoefficients[i][j] = weka.core.Utils.correlation(event1,
	// event2, numAttributes);
	// }
	// }
	// }
	// return correlationCoefficients;
	// }

	private boolean instancesCorrelated2(Instances inst, Instances newInst) throws Exception {
		if (newInst.numInstances() == 0 || inst.numInstances() == 0)
			return false;
		Instances inst1 = new Instances(inst);
		Instances inst2 = new Instances(newInst);

		String[] clusterLabels = new String[inst1.numClasses()];
		int f = 0;
		for (Enumeration c = inst1.classAttribute().enumerateValues(); c.hasMoreElements();) {

			clusterLabels[f] = ((String) c.nextElement());
			f++;
		}
		try {
			// Set the label of the first set of instances to the first label of
			// the classes ( in the header)
			for (int i = 0; i < inst1.numInstances(); i++) {
				inst1.instance(i).setClassValue(clusterLabels[0]);
			}
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			// Set the label of the second set of instances to the secind label
			// of the classes ( in the header)
			for (int i = 0; i < inst2.numInstances(); i++) {
				inst2.instance(i).setClassValue(clusterLabels[1]);
			}
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		Instances all_instances = new Instances(inst1);
		all_instances = merge(inst1, inst2);
		String[] options = new String[4];
		options[0] = "-I"; // max. iterations
		options[1] = "100";
		options[2] = "-N";
		options[3] = "2";
		EM clusterer = new EM(); // new instance of clusterer
		clusterer.setOptions(options); // new instance of clusterer
		// set the options
		// build the clusterer

		// Clustering
		String[] attribNames = new String[all_instances.numAttributes()];
		all_instances.setClassIndex(attribNames.length - 1);
		try {
			clusterer.buildClusterer(removeClass(all_instances));
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		ClusterEvaluation eval2 = new ClusterEvaluation();

		eval2.setClusterer(clusterer);

		try {
			eval2.evaluateClusterer(all_instances, "", false);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		double[] Assignments = eval2.getClusterAssignments();

		// Return the biggest
		ArrayList<Instances> subClusters = getSubClusters(all_instances, Assignments, eval2.getNumClusters());
		double[] purity = new double[subClusters.size()];
		purity = getPurity(subClusters, all_instances.numInstances());

		boolean correlated = true;
		for (int i = 0; i < purity.length; i++)
			if (purity[i] < 0.8)
				correlated = false;

		return correlated;
	}

	private double[] getPurity(ArrayList<Instances> subClusters, int n) {
		double[] perc = new double[subClusters.size()];
		double[] purity = new double[subClusters.size()];
		boolean correlated = false;
		for (int i = 0; i < subClusters.size(); i++) {
			perc[i] = (double) subClusters.get(i).numInstances() / (double) n;
			if (perc[i] <= 0.2)
				correlated = true;
			;
		}
		if (correlated) {
			for (int i = 0; i < subClusters.size(); i++) {
				purity[i] = 0.9;
			}
			return purity;
		}
		for (int i = 0; i < subClusters.size(); i++) {
			String[] result = getMajorityLabel(subClusters.get(i));
			purity[i] = Double.parseDouble(result[1]);
			purity[i] = (purity[i]) / (double) subClusters.size();
		}
		return purity;
	}

	private String processNewDatawithNoHistory(Instances newInst) throws Throwable {
		// Deal with new data and empty buffer

		String str = "";
		Instance NovelCentre = getInstancesCentre(removeClass(newInst));
		String[] MLabel = getMajorityLabel(newInst);
		str += MLabel[1] + "||" + newInst.numInstances() + "," + MLabel[0] + ",";
		str += " Drift info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
		for (int id = 0; id < driftClassID.size(); id++)
			str += "DID= " + this.driftClassID.get(id) + ",";
		String strDis = "";
		String details = "";
		boolean[] inside = new boolean[this.m_noClasses];
		boolean[] inSlack = new boolean[this.m_noClasses];
		String insideStr = "";
		String inSlackStr = "";
		// Compare the distance from new Instances to the class centre with the
		// max distance inside the class
		// ERROR>>> When the concept drift is realted to other class ( buffer
		// concept drift is diffrent from new data concept drift)

		// To decide INSIDE and Slack arrays
		double[] SZ = new double[this.m_noClasses];
		for (int i = 0; i < this.m_noClasses; i++) {
			//
			// if(
			// this.m_classesWithClusters.get(i).getLable().toLowerCase().contains("walk"))
			// SZ[i]= 900;
			// else
			// if(
			// this.m_classesWithClusters.get(i).getLable().toLowerCase().contains("sit"))
			// SZ[i]= 3500;
			// else
			SZ[i] = this.slackSize[i];
			inside[i] = false;
			inSlack[i] = false;
			double distanceDiff = CalculateED(NovelCentre, this.m_classesWithClusters.get(i).m_classCentre);
			strDis += SZ[i] + "," + distanceDiff + ",";
			if (distanceDiff < this.d_max[i])
				inside[i] = true;
			else if (distanceDiff - this.d_max[i] < SZ[i])
				inSlack[i] = true;
			insideStr += inside[i] + ",";
			inSlackStr += inSlack[i] + ",";
		}

		str += insideStr + inSlackStr;
		// If it is inside 2 classes, we choose the closest
		int hitInsideID = IDofInsideHit(inside, NovelCentre);

		// If it is inside the slack, we choose the closest
		int inSlackID = IDofInsideHit(inSlack, NovelCentre);
		str += hitInsideID + "," + inSlackID + ",";
		// Inside any?

		// Case 1.1: Is inside and the buffer is empty
		if (hitInsideID != -1) {
			details += "Case 1.1,  New chunk with size recognised as inside the class " + hitInsideID + ",";
			casesCounter[1][1]++;
			if (this.m_classesWithClusters.get(hitInsideID).m_classID == -1) {
				this.accumalateNBClass(newInst, hitInsideID);
				reNovelFlag = true;
			} else
				m_predLabel = m_Oldlabels[hitInsideID];
			dataWarehouse = new Instances(newInst);
			this.justPredicted = new Instances(newInst);
			str += strDis;
			str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
			for (int id = 0; id < driftClassID.size(); id++)
				str += "DID= " + this.driftClassID.get(id) + ",";
			str += details + "\n";
			return str;

		}
		// Casre 1.2: In Slack

		if (inSlackID != -1) {

			this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
			for (int i = 0; i < this.m_noClasses; i++) {
				if (inSlack[i]) {
					if (!this.driftClassID.contains(i))
						this.driftClassID.add(i);

				}
			}
			this.suspectedDrift = true;
			m_predLabel = "";
			details += "Case 1.2,  New data added to buffer as suspected drift ,";
			this.justPredicted = new Instances(newInst, 0);
			casesCounter[1][2]++;

			str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
			for (int id = 0; id < driftClassID.size(); id++)
				str += "DID= " + this.driftClassID.get(id) + ",";
			str += details + "\n";
			return str;

		}
		// Not known ( Inside or in slack of any)
		this.suspNovelInstances = merge(this.suspNovelInstances, newInst);
		details += "Case 1.3,  New data added to buffer as unknown" + ",";
		casesCounter[1][3]++;
		m_predLabel = "";
		this.justPredicted = new Instances(newInst, 0);
		str += " Drift Info: " + this.suspNovelInstances.numInstances() + "," + this.suspectedDrift + ",";
		for (int id = 0; id < driftClassID.size(); id++)
			str += "DID= " + this.driftClassID.get(id) + ",";
		str += details + "\n";
		return str;

	}

	int nbSize = 1;
	// accumalate new norn class with new recognised instances
	private BLM CWSCModelOld = null;

	private void accumalateNBClass(Instances newInst, int i) throws Exception {

		ClassWSubClusters prevObj = (ClassWSubClusters) this.m_classesWithClusters.get(i).clone();
		Instances currentInstances = new Instances(prevObj.getInstances());
		currentInstances = merge(currentInstances, newInst);

		if (currentInstances.numInstances() > nbSize * 2000 && prevObj.m_totalsize < 10000) {
			nbSize++;
			ReformNBClass(newInst, i);
			return;
		}

		updateNBClass(newInst, i);
	}

	private void updateNBClass(Instances Inst, int ID) throws Exception {

		ClassWSubClusters obj = (ClassWSubClusters) this.m_classesWithClusters.get(ID);
		// Check
		Instances currentInstances = new Instances(obj.getInstances());
		currentInstances = merge(currentInstances, Inst);
		obj.setInstances(currentInstances);
		Instances newInst = new Instances(removeClass(Inst));
		int pos = chooseSubClusterToUpdate(obj, newInst);

		Instance oldCentre = (Instance) obj.m_centres[pos].copy();
		Instance centre_old = obj.m_centres[pos];
		obj.m_centres[pos] = updateCentre(obj.m_size[pos], newInst, oldCentre);
		centerMovement = CalculateED(centre_old, obj.m_centres[pos]);
		this.m_SysCentre = getSysCentre(this.m_classesWithClusters);
		double oldBoundry = obj.m_classBoundry[pos];
		// ERROR
		obj.m_classBoundry[pos] = updateBoundry((Instance) obj.m_centres[pos].copy(), newInst, oldBoundry);
		obj.m_size[pos] += newInst.numInstances();
		obj.m_totalsize += newInst.numInstances();
		this.m_sysCapacity += newInst.numInstances();
		// double temp= CalculateED(obj.m_centres[0], obj.m_centres[1]);

		// if( temp> this.d_max[ID])
		// this.d_max[ID]= temp;

		// obj.m_VDL= updateVDL(obj);
		// obj.m_maxVDL= getMaxLoss(obj.m_VDL);
		// obj.m_minGravF= sysMinGF(ID);

		this.novelClassCentre = null;

		this.suspNovelInstances.delete();
		this.suspectedDrift = false;

	}

	private void ReformNBClass(Instances newInst, int i) throws Exception {

		// remove previous obj and add an updated one with updated
		// charachtaristices( Instances)
		ClassWSubClusters prevObj = (ClassWSubClusters) this.m_classesWithClusters.get(i).clone();

		// Check
		Instances currentInstances = new Instances(prevObj.getInstances());

		currentInstances = merge(currentInstances, newInst);
		this.m_classesWithClusters.remove(i);
		ArrayList<Instances> classSubClusters = new ArrayList<Instances>();
		classSubClusters = initSubClusters(currentInstances);

		// Updated for memory isssues

		IntCWSCClass intClass = new IntCWSCClass(classSubClusters, prevObj.label, -1);
		ClassWSubClusters CWSC = new ClassWSubClusters(intClass.m_centres, intClass.m_size, intClass.m_SD,
				intClass.m_classBoundry, intClass.m_averageDistance, intClass.NoOfSubClusters, intClass.label,
				intClass.m_classID, intClass.m_classSD, intClass.m_farthest_dis, intClass.m_globalAvDistance,
				intClass.m_globalBoundry, intClass.m_gravForce, intClass.m_maxVDL, intClass.m_minGravF, intClass.m_size,
				intClass.m_totalsize, intClass.m_VDL, intClass.m_classCentre);
		// Check
		CWSC.setInstances(currentInstances);
		this.m_classesWithClusters.add(CWSC);
		this.m_sysCapacity += newInst.numInstances();
		this.m_SysCentre = getSysCentre(this.m_classesWithClusters);
		this.novelClassCentre = null;
		this.d_max[this.m_noClasses - 1] = findNovelMaxDistance();
		this.suspNovelInstances.delete();
		this.suspectedDrift = false;

	}

	private void addNovelClass(Instances suspNovelInstances, String label) throws Exception {

		// If it is FP then return with no action
		ALStats = "";
		this.CWSCModelOld = new BLM(this);
		boolean FP_Flag = false;
		for (int i = 0; i < this.m_Oldlabels.length; i++) {
			if (this.m_Oldlabels[i].contains(label))
				FP_Flag = true;

		}
		if (FP_Flag) {
			ALcases[0]++;
			ALStats += " \nFP ---- Expandnding cluster with label " + label + "\n";
			incrementD();
			ALStats += this.expandCluster(label);
			return;
		}

		// If it is recurrent class, accumlate NBClass
		for (int i = 0; i < this.m_classesWithClusters.size(); i++) {
			if (this.m_classesWithClusters.get(i).label.toLowerCase().contains(label.toLowerCase())
					&& this.m_classesWithClusters.get(i).m_classID == -1) {
				ALcases[1]++;
				ALStats += "\nAccumlate on novel class \n";
				accumalateNBClass(suspNovelInstances, i);
				return;
			}
		}
		ALStats += "\n create novel class \n";
		ALcases[2]++;
		ArrayList<Instances> classSubClusters = new ArrayList<Instances>();
		classSubClusters = initSubClusters(suspNovelInstances);
		// ClassWSubClusters CWSC = new ClassWSubClusters(classSubClusters,
		// label,
		// -1);
		IntCWSCClass intClass = new IntCWSCClass(classSubClusters, label, -1);
		ClassWSubClusters CWSC = new ClassWSubClusters(intClass.m_centres, intClass.m_size, intClass.m_SD,
				intClass.m_classBoundry, intClass.m_averageDistance, intClass.NoOfSubClusters, intClass.label,
				intClass.m_classID, intClass.m_classSD, intClass.m_farthest_dis, intClass.m_globalAvDistance,
				intClass.m_globalBoundry, intClass.m_gravForce, intClass.m_maxVDL, intClass.m_minGravF, intClass.m_size,
				intClass.m_totalsize, intClass.m_VDL, intClass.m_classCentre);
		// Check
		CWSC.setInstances(suspNovelInstances);
		this.m_classesWithClusters.add(CWSC);
		this.m_noClasses++;
		String[] newLabels = new String[this.m_labels.length + 1];
		double[] tempSZ = new double[this.slackSize.length + 1];
		double[] temp_dmax = new double[this.d_max.length + 1];
		for (int i = 0; i < this.m_labels.length; i++) {
			newLabels[i] = this.m_labels[i];
			tempSZ[i] = this.slackSize[i];
			temp_dmax[i] = this.d_max[i];

			if (this.m_labels[i] == label)
				FP_Flag = true;
		}
		// if( label.toLowerCase().contains("sit"))
		// tempSZ[this.slackSize.length]= 3500;
		tempSZ[this.slackSize.length] = this.defaultSlackSize;
		newLabels[this.m_labels.length] = label;
		temp_dmax[this.d_max.length] = this.findNovelMaxDistance();
		this.slackSize = tempSZ.clone();
		this.m_labels = newLabels.clone();
		this.d_max = temp_dmax.clone();

		this.m_sysCapacity += suspNovelInstances.numInstances();
		this.m_SysCentre = getSysCentre(this.m_classesWithClusters);
		// this.m_FathestDis= getFathestDistance(this.m_classesWithClusters);

		// Novel observation
		this.novelClassCentre = null;

	}

	private ArrayList<Instances> initSubClusters(Instances instances) throws Exception {

		Instances iniInst = new Instances(instances);
		String[] options = new String[4];
		options[0] = "-I"; // max. iterations
		options[1] = "100";
		options[2] = "-N";
		options[3] = "2";
		SimpleKMeans clusterer = new SimpleKMeans(); // new instance of
														// clusterer
		clusterer.setOptions(options); // set the options
		// build the clusterer

		// Clustering
		// String [] attribNames = new String [iniInst.numAttributes()];
		// iniInst.setClassIndex(attribNames.length - 1);
		try {
			clusterer.buildClusterer(removeClass(iniInst));
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		ClusterEvaluation eval2 = new ClusterEvaluation();

		eval2.setClusterer(clusterer);

		try {
			eval2.evaluateClusterer(iniInst, "", false);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		double[] Assignments = eval2.getClusterAssignments();

		// Return the biggest
		ArrayList<Instances> subClusters = getSubClusters(iniInst, Assignments, eval2.getNumClusters());

		return subClusters;

	}

	private ArrayList<Instances> getSubClusters(Instances iniInst, double[] assignments, int k) {
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

	private double OldDensityfn(Instances n, Instance cntr) {
		double den = 0, avdist = 0;
		for (int i = 0; i < n.numInstances(); i++) {
			avdist += CalculateED(n.instance(i), cntr);
		}
		avdist /= (double) n.numInstances();

		den = (double) n.numInstances() / avdist;
		return den;
	}

	private Instance accumalateCentre(Instance newCentre, Instances newInsances) {
		if (this.novelClassCentre == null)
			return newCentre;

		// accumalated point
		Instance updatedMean = (Instance) this.novelClassCentre.copy();
		double n = newInsances.numInstances();
		Instance nIns = (Instance) newInsances.firstInstance().copy();
		double[] mean = new double[nIns.numAttributes()];
		for (int i = 0; i < newInsances.numInstances(); i++) {
			n++;
			nIns = newInsances.instance(i);
			for (int j = 0; j < nIns.numAttributes(); j++) {
				mean[j] = 0;
				double delta = nIns.value(j) - updatedMean.value(j);
				mean[j] = updatedMean.value(j) + (delta / n);
				updatedMean.setValue(j, mean[j]);
			}
		}
		return updatedMean;

	}

	public double getMaxDistance(int v) {
		// TODO Auto-generated method stub
		return this.d_max[v];
	}

	public void setTestInstancesHeader(Instances inst) {
		inst.setClassIndex(inst.numAttributes() - 1);
		this.suspNovelInstances = new Instances(inst, 0);
		this.dataWarehouse = new Instances(inst, 0);
		this.justPredicted = new Instances(inst, 0);
		this.CWSCModelOld = new BLM(this);

	}

	double centerMovement = 0;
	private int incCounter = 0;
	private int actUnknownCounter = 0;

	public String getCentreMovement() {
		// TODO Auto-generated method stub
		return Double.toString(centerMovement);
	}

	public void resetAll() {
		this.accumlate = false;
		this.reset = false;
		this.cutoff_index = -1;
		this.suspectedDrift = false;
		this.driftClassID.clear();

	}

	public void lastSegDecision() {
		double distance = 0;
		int ID = -100;

		if (accumlate) {
			dataWarehouse = new Instances(this.suspNovelInstances, cutoff_index,
					this.suspNovelInstances.numInstances() - cutoff_index);
		} else
			dataWarehouse = new Instances(this.suspNovelInstances);
		if (this.driftClassID.size() == 1) {
			ID = this.driftClassID.get(0);
		}

		else if (this.driftClassID.size() > 1) {
			// Choose closest
			for (int i = 0; i < this.driftClassID.size(); i++) {
				double temp = CalculateED(getInstancesCentre(removeClass(this.suspNovelInstances)),
						this.m_classesWithClusters.get(driftClassID.get(i)).m_classCentre);
				if (temp > distance) {
					distance = temp;
					ID = driftClassID.get(i);
				}
			}

		} else {
			m_predLabel = "Unknown";
			return;
		}
		if (this.m_classesWithClusters.get(ID).m_classID == -1)
			reNovelFlag = true;
		else
			m_predLabel = m_Oldlabels[ID];
		return;

	}

	public String updateModelFn(String newLabel) throws Exception {

		String s = "";

		if (this.isNovelClassCreated()) {
			for (int i = 0; i < this.m_classesWithClusters.size(); i++) {
				if (this.m_classesWithClusters.get(i).label.toLowerCase().contains(newLabel.toLowerCase())
						&& this.m_classesWithClusters.get(i).m_classID == -1) {
					ClassWSubClusters prevObj = (ClassWSubClusters) this.m_classesWithClusters.get(i).clone();
					Instances currentInstances = new Instances(prevObj.getInstances());
					currentInstances = merge(dataWarehouse, currentInstances);
					if (currentInstances.numInstances() > nbSize * 2000 && prevObj.m_totalsize < 10000) {
						nbSize++;
						ReformNBClass(dataWarehouse, i);
						s += "Reform NB class\n";
					} else {
						updateNBClass(dataWarehouse, i);
						s += "update NB class\n";
					}
					for (int md = 0; md < this.getNumberOfClasses(); md++)
						s += " Max Distance= " + this.getMaxDistance(md) + " ";
					if (unkFlag)
						unknownCases[2]++;
					return s;
				}
			}
		}
		this.addNovelClass(dataWarehouse, newLabel);
		if (unkFlag)
			unknownCases[3]++;
		s += "\n  NB Class Added\n";
		for (int md = 0; md < this.getNumberOfClasses(); md++)
			s += " Max Distance= " + this.getMaxDistance(md) + " ";

		s += "\n";
		return s;
	}

	public String updateModelConceptEvolution(NovelPredection pred) throws Exception {
		String s = "";
		double purity = Double.parseDouble(pred.getClassTrueLabel()[1]);
		purity /= pred.getClassSize();

		if (pred.getNoveltyType().contains("Fn")) {
			if (purity < 0.8) {
				s = "No-Update, purity= " + purity + ",\n";

				return s;
			}
			s += " Update-Model \n";
			s += this.updateModelFn(pred.getClassTrueLabel()[0]);
			s += this.shrinkCluster(pred.getPredictedLabel());

			// s+= this.expandCluster(pred.getClassTrueLabel()[0]);
			return s;
		}

		if (pred.getNoveltyType().contains("Fp")) {
			if (purity < 0.8) {
				s = "No-Update, purity= " + purity + "\n";

				return s;

			}
			s += " Update-Model \n";
			incrementD();
			// s+= this.shrinkCluster(pred.getPredictedLabel());
			s += this.expandCluster(pred.getClassTrueLabel()[0]);
			return s;
		}
		return s;
	}

	public String updateModelConceptEvolution_Unknown(Instances inst, NovelPredection pred) throws Exception {
		String s = "";
		unkFlag = true;
		double purity = Double.parseDouble(pred.getClassTrueLabel()[1]);
		purity /= pred.getClassSize();

		if (purity < 0.8) {
			s = "No-Update, purity= " + purity + ",\n";
			incrementunknownActive();
			unknownCases[0]++;
			unkFlag = false;
			return s;
		}

		s += " Update-Model \n";
		s += this.updateCSWSCModel(inst, pred.getClassTrueLabel()[0]);
		unkFlag = false;
		return s;

	}

	private void incrementD() {
		this.incCounter++;

	}

	private void incrementunknownActive() {

		this.actUnknownCounter++;

	}

	private String expandCluster(String predictedLabel) {
		String s = "";
		int classID = -1;
		for (int i = 0; i < this.m_labels.length; i++) {
			if (this.m_labels[i].equals(predictedLabel.trim())) {
				classID = i;
				break;
			}
		}
		Instance novelInsCntr = getInstancesCentre(removeClass(dataWarehouse));

		if (classID != -1) {
			double distance = CalculateED(novelInsCntr, this.m_classesWithClusters.get(classID).m_classCentre);
			s += " Class " + classID + " Old d_max= " + d_max[classID] + " Distance= " + distance + "		";
			double d = slackSize[classID];
			if (d_max[classID] + slackSize[classID] < distance) {
				slackSize[classID] = distance;
				s += " new Slack (In Slack)= " + slackSize[classID] + "			\n";
				return s;
			}
			s += "In Slack already ";
		}
		s += " No-Change \n";
		return s;
	}

	private String shrinkCluster(String predictedLabel) {
		String s = "";
		int classID = -1;
		for (int i = 0; i < this.m_labels.length; i++) {
			if (this.m_labels[i].equals(predictedLabel.trim())) {
				classID = i;
				break;
			}
		}
		Instance novelInsCntr = getInstancesCentre(removeClass(dataWarehouse));

		if (classID != -1) {
			double distance = CalculateED(novelInsCntr, this.m_classesWithClusters.get(classID).m_classCentre);
			s += " Old d_max= " + d_max[classID] + " Distance= " + distance + "		";
			// inside
			if (this.d_max[classID] > distance) {
				this.d_max[classID] = distance;
				s += " new d_max ( inside) = " + this.d_max[classID] + "			\n";
				incrementD();
				return s;
			}
			double d = slackSize[classID];
			if (d_max[classID] + slackSize[classID] > distance) {
				slackSize[classID] = distance;
				s += " new Slack (In Slack)= " + slackSize[classID] + "			\n";
				incrementD();
				return s;
			}
			s += "Outside ";
		}
		s += " No change \n";
		return s;
	}

	public int[][] getCasesCounter() {
		return this.casesCounter;
	}

	public void setNoveltSlack(double d) {
		this.defaultSlackSize = d;

	}

	public int getActiveUnknowonCounter() {
		return actUnknownCounter;
	}

	public int getIncCounter() {
		return incCounter;
	}

	public int getNumClasse() {
		// TODO Auto-generated method stub
		return m_noClasses;
	}

	public int[] actLearningCases() {
		return ALcases;
	}

	public int actLearningRate() {
		int n = 0;
		for (int i = 0; i < 3; i++)
			n += ALcases[i];
		return n;

	}

	public int[] unKnownStats() {
		return unknownCases;

	}
	// public SubClustersModel clearModel() {
	// for( int i=0; i< this.m_classesWithClusters.size(); i++)
	// return null;
	// }

	public boolean getUpateFlag() {
		return this.updFlag;
	}

	public boolean getvalidFlag() {
		return this.validFlag;
	}

	public void setParameters(HashMap<String, String> parameters) {
		// TODO Auto-generated method stub
		System.out.println("Reading parameters...");
		System.out.println("-------------");

		double novelSlackSize = Double.parseDouble(parameters.get("Novel_Slack"));
		System.out.println("Novel Slack: " + novelSlackSize);
		double movement = Double.parseDouble(parameters.get("Movement"));
		System.out.println("Movement : " + movement);
		int stable_size = Integer.parseInt(parameters.get("Stable_Size"));
		System.out.println("Stable Size : " + stable_size);
		boolean b = Boolean.parseBoolean(parameters.get("Buffer_Flag"));
		System.out.println("Buffer size falg: " + b);
		double at = Double.parseDouble(parameters.get("Away_Threshold"));
		System.out.println("Away temp : " + at);
		boolean JPLF = Boolean.parseBoolean(parameters.get("JP_Layer"));
		System.out.println("JPlayer: " + JPLF);
		boolean updateFlag = Boolean.parseBoolean(parameters.get("Update_Flag"));
		System.out.println("Update_Flag: " + updateFlag);
		boolean validationFlag = Boolean.parseBoolean(parameters.get("Validate_Flag"));
		System.out.println("Validation Flag: " + validationFlag);

		// Set different slack sizes
		String slackTxt = (String) parameters.get("Slacks");
		if (!slackTxt.contains(",")) {
			double slack_size = Double.parseDouble(slackTxt);
			this.setSlackSize(slack_size);
			System.out.println("Default slack sizes is: " + slack_size);
		} else {
			if (compatible(slackTxt, this)) {
				double[] slackSize = new double[this.getNumberOfClasses()];
				String[] SlackString = new String[this.getNumberOfClasses()];
				SlackString = slackTxt.split(",");
				for (int ss = 0; ss < SlackString.length; ss++) {
					slackSize[ss] = Double.parseDouble(SlackString[ss]);
					System.out.println("Slack Size of Class: " + ss + " is " + slackSize[ss] + " With Label: "
							+ this.getLabels()[ss]);
				}
				this.setSlackSize(slackSize);
			}

			else {

				this.setSlackSize(defaultSlackSize);
				System.out.println("Error in slack Size, set all to the default size");
			}

		}
		this.setUpadteFlag(updateFlag);
		this.setValidationFlag(validationFlag);
		this.setNoveltSlack(defaultSlackSize);
		this.setMovementThreshold(movement);
		this.setBufferFlag(b);
		this.setJPLFlag(JPLF);
		this.setawaytemp(at);
		this.setStableClusterSize(stable_size);

	}

	public void setValidationFlag(boolean validationFlag) {
		this.validFlag = validationFlag;

	}

	private void setUpadteFlag(boolean updateFlag) {
		this.updFlag = updateFlag;

	}

	private static boolean compatible(String slackTxt, BLM cWSCModelCpy) {

		int slackCount = 1;
		while (slackTxt.contains(",")) {
			int endIndex = slackTxt.indexOf(",");
			slackTxt = slackTxt.substring(endIndex + 1);
			slackCount++;
		}

		if (cWSCModelCpy.getLabels().length == slackCount)
			return true;
		return false;
	}

	public static Instances merge(Instances data1, Instances data2) throws Exception {
		// Check where are the string attributes
		int asize = data1.numAttributes();
		boolean strings_pos[] = new boolean[asize];
		for (int i = 0; i < asize; i++) {
			Attribute att = data1.attribute(i);
			strings_pos[i] = ((att.type() == Attribute.STRING) || (att.type() == Attribute.NOMINAL));
		}

		// Create a new dataset
		Instances dest = new Instances(data1);
		dest.setRelationName(data1.relationName() + "+" + data2.relationName());

		DataSource source = new DataSource(data2);
		Instances instances = source.getStructure();
		Instance instance = null;
		while (source.hasMoreElements(instances)) {
			instance = source.nextElement(instances);
			dest.add(instance);

			// Copy string attributes
			for (int i = 0; i < asize; i++) {
				if (strings_pos[i]) {
					dest.instance(dest.numInstances() - 1).setValue(i, instance.stringValue(i));
				}
			}
		}

		return dest;
	}

}
