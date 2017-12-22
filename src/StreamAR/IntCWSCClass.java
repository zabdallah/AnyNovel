package StreamAR;

import java.io.Serializable;
import java.util.ArrayList;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
public class IntCWSCClass implements Cloneable , Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1695784640371321039L;
	public double m_minGravF;
	public int m_totalsize;
	public double m_globalBoundry;
	public double m_globalAvDistance;
	public int m_classID;
	public int NoOfSubClusters;
	public Instance []m_centres;
	public int []m_size;
	public Instance[] m_SD;
	public String label; 
	public double[] m_classBoundry;
	public double[] m_averageDistance;
	public int[] weights;
	public Instance m_classCentre;
	public Instance m_classSD;
	public double m_farthest_dis;
	public double m_maxVDL;
	public double[] m_VDL;
	public double[][]m_gravForce;
	public Instances train_Instances;
	
public  IntCWSCClass (ArrayList<Instances> subClusters, String classLabel, int ID)
{
	NoOfSubClusters= subClusters.size();
	m_centres= setCentres(subClusters);
	m_size= setSizes(subClusters);
	m_SD= setSD(subClusters);
	m_classBoundry= setBoundries(subClusters);
	m_averageDistance= setAverageDistances(subClusters);
	label= classLabel;
	m_classID= ID;
	for( int i=0; i<m_size.length;i++)
		m_totalsize+= m_size[i];
	
	Instances m_allInstances= getAllInsatnces(subClusters);
	
	m_classCentre= getClassCentre();
//	m_classCentre=getClassCentre(m_allInstances) ;
	
	m_globalBoundry= getGlobalBoundry();
//	m_globalBoundry= getGlobalBoundry(m_allInstances);
	
//	m_globalAvDistance= getGlobalAvDistance();
	m_globalAvDistance= getGlobalAvDistance(m_allInstances);
	
	m_classSD= getGlabalSD(m_allInstances);
	
	this.m_VDL= new double[this.NoOfSubClusters];
	this.m_VDL= getVirtualDensityLoss();
	this.m_maxVDL=getmaxVDL(this.m_VDL);
	this.m_farthest_dis= getDistanceThreshold(this.m_centres, this.m_classCentre);
	this.m_minGravF= getMinGravForce();
	this.m_gravForce= new double[this.NoOfSubClusters][this.NoOfSubClusters];
	this.m_gravForce= getGravForce();
	subClusters.clear(); 
	System.gc(); 
	
}
private double[][] getGravForce() {
	
	double[][] GF= new double[this.NoOfSubClusters][this.NoOfSubClusters];
	
	for( int j=0; j< this.NoOfSubClusters; j++){
		
		for( int k=0; k< this.NoOfSubClusters; k++){
			if( k!=j){
				GF[j][k]= gravetationForce(this.m_size[k], this.m_size[j],
				(Instance)this.m_centres[k].copy(),(Instance)this.m_centres[j].copy());
			}
	}
	}
return GF;
}

private double getMinGravForce() {

				double temp=0;
				double minGF= Double.MAX_VALUE;
				for( int j=0; j< this.NoOfSubClusters; j++){
					for( int k=0; k< this.NoOfSubClusters; k++){
						if( k!=j){
					temp= gravetationForce(this.m_size[k], this.m_size[j],
							(Instance)this.m_centres[k].copy(),(Instance)this.m_centres[j].copy());
					if( temp< minGF)
						minGF=temp;
					}
				}
				}
		return minGF;
}

private double gravetationForce(int sizei, int sizej, Instance m_centres2, Instance m_centres3) {
	double distance;
//	  EuclideanDistance ED= new EuclideanDistance(this.m_centres);
	  distance= CalculateED(m_centres2, m_centres3);
	  double result= (double)(sizei*sizej)/ ( distance*distance);
	 return result;
}

private double getmaxVDL(double[] loss) {
	// TODO Auto-generated method stub
	double max=Double.MAX_VALUE;
	for ( int i=0; i<loss.length;i++)
		if(loss[i]<max )
			max=loss[i];
	return max;
}

private double getDistanceThreshold(Instance[] centres, Instance classCentre) {
	//Get the farthest distance between any of the subclusters exist into the class
	double furthestDistance=-1;
	double temp=0;
	for(int i=0; i< centres.length;i++ )
	{
				temp= CalculateED(centres[i], classCentre);
				if( temp> furthestDistance) 
					furthestDistance= temp;
			
		
	}
	
	return furthestDistance;
}
private double[] getVirtualDensityLoss() {

double[] loss= new double[this.NoOfSubClusters];
double orgDensity=  densityFunction((double)this.m_totalsize, this.m_globalBoundry, 3);
for (int j = 0; j < this.NoOfSubClusters;  j++) {
	int newsize= this.m_totalsize - this.m_size[j];	
	double newRadius=getNewRadius(j);
	double newDensity= densityFunction((double)newsize, newRadius,3) ;
	loss[j]=newDensity-orgDensity  ; 
			
	}
		return loss;
		}
	
	private double getNewRadius(int subClusterID) {
		
			Instance newCentre= getSubClustersCentre(subClusterID);
			double boundry=0; 
			double temp=0;
			for ( int i=0; i< this.m_centres.length;i++){
				if( i!=subClusterID){
					temp= CalculateED(newCentre, this.m_centres[i])+ this.m_classBoundry[i];
					if(temp> boundry) boundry=temp;
				}
			}
			return boundry;
		}


	private Instance getSubClustersCentre(int exCluster) {
		// Calculate Class centre with excluding sub-cluster
		
		Instance middle= (Instance) this.m_centres[0].copy();
		double [] middleArray= new double [middle.numAttributes()];
		int n= this.m_totalsize- this.m_size[exCluster];
		
		for( int i=0; i< middle.numAttributes();i++){
			for( int j=0; j< this.NoOfSubClusters;j++)
				if( j!=exCluster)
					middleArray[i]+= this.m_centres[j].value(i)*this.m_size[j];
			middleArray[i]/=( double) n;
			
		}
		for( int i=0; i< middle.numAttributes();i++){
		middle.setValue(i, middleArray[i]);
		}
		return middle;
	}
	private Instance[] setSD(ArrayList<Instances> subClusters) {
		Instance[] SDInstances= new Instance[this.NoOfSubClusters];
		for( int classID=0; classID<this.NoOfSubClusters;classID++){
			
			
			Instances m_instances= new Instances(removeClass(subClusters.get(classID)));
			SDInstances[classID]= (Instance) m_instances.firstInstance().copy();
			double SD[]= new double[m_instances.numAttributes()];
//			Instances microCluster= createMicroCluster(m_instances,classID,  3000);
			for( int i=0; i<m_instances.numAttributes();i++)
				SD[i]=m_instances.variance(i);
		
			for( int j=0; j< SD.length;j++){
				SD[j]= Math.sqrt(SD[j]);
				SDInstances[classID].setValue(j ,SD[j]);
			}
		
		}
		//inst.setClassValue(-1);
			return SDInstances;
	
	}

	private int[] setSizes(ArrayList<Instances> subClusters) {
		
		
		int[] count= new int[ this.NoOfSubClusters];
		for( int i=0; i< this.NoOfSubClusters;i++){
		Instances m_instances= new Instances(subClusters.get(i));
		 count[i]=m_instances.numInstances();
		}
		return count;
	}

	private Instance[] setCentres(ArrayList<Instances> subClusters) {
		
		Instances new_Instances= new Instances(removeClass(subClusters.get(0)));
		Instance[] centres= new Instance[this.NoOfSubClusters];
//		centresArr= new double[m_noClasses][centres.numAttributes()];
		
		 for( int classID=0; classID<this.NoOfSubClusters;classID++){
		
			 centres[classID]= (Instance)new_Instances.firstInstance().copy(); 
			Instances m_instances= new Instances(removeClass(subClusters.get(classID)));
			int m_numAtt= m_instances.numAttributes();
			double[] mean= new double[m_numAtt];
			 for( int i=0; i< m_numAtt;i++){
				 mean[i]= m_instances.meanOrMode(i);
			 }
			
			for( int i=0;i<m_numAtt;i++ )
				centres[classID].setValue(i,mean[i]);
		 }
		
		return centres;
	
	}
	private double[] setAverageDistances(ArrayList<Instances> subClusters) {
		double[] AvDistanceArr = new double[subClusters.size()];
		for( int classID=0; classID< subClusters.size();classID++)
		{
		
		Instances m_instances= new Instances(subClusters.get(classID));
		Instance cntrInst= (Instance)this.m_centres[classID].copy();
		double distance=0;
		for( int i=0; i<m_instances.numInstances(); i++){
			distance+= CalculateED(cntrInst, m_instances.instance(i));	
		}
		distance/= m_instances.numInstances();
		
		AvDistanceArr[classID]=distance;
		}
		return AvDistanceArr;

}
	
	
private double[] setBoundries(ArrayList<Instances> subClusters) {
		
		double[] maxDistanceArr = new double[subClusters.size()];
		
		for( int classID=0; classID< subClusters.size();classID++)
		{
		
		Instances m_instances= new Instances(subClusters.get(classID));
//		EuclideanDistance ED= new EuclideanDistance(m_instances);
//		ED.setDontNormalize(true);
		double maxDistance=0;
		Instance cntrInst= getClassCentre(m_instances);
		double distance=0;
		for( int i=0; i<m_instances.numInstances(); i++){
			distance= CalculateED(cntrInst, m_instances.instance(i));
//			distance= ED.distance(cntrInst,  m_instances.get(i));
		
			if( distance> maxDistance)
				maxDistance=distance;
		}
	
		maxDistanceArr[classID]=maxDistance;
		}
		return maxDistanceArr;
		
	}

private Instances getAllInsatnces(ArrayList<Instances> subClusters) {
	Instances allInstances= new Instances(removeClass(subClusters.get(0)));
	for( int i=1; i<this.NoOfSubClusters;i++){
		
		Instances newInstances= new Instances(removeClass(subClusters.get(i)));
		for( int j=0; j<newInstances.numInstances();j++)
			allInstances.add(newInstances.instance(j));
	} 

	return allInstances;
}

private Instance getGlabalSD(Instances allInstances) {

	double SD[]= new double[allInstances.numAttributes()];
	Instance SDInstance= (Instance) allInstances.firstInstance().copy();
	for( int i=0; i<allInstances.numAttributes();i++)
			SD[i]=allInstances.variance(i);	
	
	for( int j=0; j< SD.length;j++){
		SD[j]= Math.sqrt(SD[j]);
		SDInstance.setValue(j ,SD[j]);
	}
	//inst.setClassValue(-1);
		return SDInstance;
}
//private double getGlobalAvDistance() {
//
//	double Average=0; 
//	
//	for( int i=0; i< this.NoOfSubClusters;i++){
//		Average+=this.m_averageDistance[i]*this.m_size[i];
//	}
//	Average/= this.m_totalsize;
//	
//	return Average;
//}

private Instance getClassCentre() {
	Instance middle= (Instance) this.m_centres[0].copy();
	double [] middleArray= new double [middle.numAttributes()];
	for( int i=0; i< middle.numAttributes();i++){
		for( int j=0; j< this.NoOfSubClusters;j++)
				middleArray[i]+= this.m_centres[j].value(i)*this.m_size[j];
		middleArray[i]/=( double) this.m_totalsize;
		
	}
	for( int i=0; i< middle.numAttributes();i++){
		middle.setValue(i, middleArray[i]);
	}
	return middle;
	
}	
private double getGlobalBoundry() {
	
	double temp=0; 
	double furthest=0; 
	
	for( int i=0; i< this.NoOfSubClusters;i++){
		temp= CalculateED(m_classCentre, m_centres[i]) + this.m_classBoundry[i];
		if(temp> furthest)
			furthest=temp;
	}
	return furthest;
}
private double getGlobalAvDistance(Instances allInstances) {

	Instance cntrInst= (Instance)this.m_classCentre.copy();
	double distance=0;
	for( int i=0; i<allInstances.numInstances(); i++){
		distance+= CalculateED(cntrInst, allInstances.instance(i));
		
	}
	return distance/this.m_totalsize;
}

private double CalculateED(Instance first,Instance second ){
	
	
	 double dist = 0.0;

	    for (int i = 0; i < first.numAttributes(); i++) {
	        double x = first.value(i);
	        double y = second.value(i);

	        if (Double.isNaN(x) || Double.isNaN(y)) {
	            continue; // Mark missing attributes ('?') as NaN.
	        }

	        dist += (x-y)*(x-y);
	    }

	    return Math.sqrt(dist);
	}
private Instances removeClass(Instances inst) {
    Remove af = new Remove();
    Instances retI = null;
    
    try {
      if (inst.classIndex() < 0) {
	retI = inst;
      } else {
	af.setAttributeIndices(""+(inst.classIndex()+1));
	af.setInvertSelection(false);
	af.setInputFormat(inst);
	retI = Filter.useFilter(inst, af);
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    return retI;
  }
private Instance getClassCentre(Instances allInstances) {
	
	
	Instance middle= (Instance) this.m_centres[0].copy();
	double [] middleArray= new double [middle.numAttributes()];
	for( int i=0; i< middle.numAttributes();i++){
			middleArray[i]=allInstances.meanOrMode(i);
	}
	for( int i=0; i< middle.numAttributes();i++){
	middle.setValue(i, middleArray[i]);
	}
	return middle;
}
private double densityFunction(double size, double radius, int d) 
{ 
	
	if( size==0)
		 return -1;
	 double v= (4/3)* Math.PI*  Math.pow(radius, d);
	 return size/v; 
}
}
