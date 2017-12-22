package StreamAR;
import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;

public class ClassWSubClusters implements Cloneable , Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 5842023170898806306L;
	protected double m_minGravF;
	protected int m_totalsize;
	protected double m_globalBoundry;
	protected double m_globalAvDistance;
	protected int m_classID;
	protected int NoOfSubClusters;
	protected Instance []m_centres;
	protected int []m_size;
	protected Instance[] m_SD;
	protected String label; 
	public double[] m_classBoundry;
	protected double[] m_averageDistance;
	protected int[] weights;
	protected Instance m_classCentre;
	protected Instance m_classSD;
	protected double m_farthest_dis;
	protected double m_maxVDL;
	protected double[] m_VDL;
	protected double[][]m_gravForce;
	protected Instances train_Instances;
//	
//	public ClassWSubClusters(ArrayList<Instances> subClusters, String classLabel, int ID)
//	{
//		NoOfSubClusters= subClusters.size();
//		m_centres= setCentres(subClusters);
//		m_size= setSizes(subClusters);
//		m_SD= setSD(subClusters);
//		m_classBoundry= setBoundries(subClusters);
//		m_averageDistance= setAverageDistances(subClusters);
//		label= classLabel;
//		m_classID= ID;
//		for( int i=0; i<m_size.length;i++)
//			m_totalsize+= m_size[i];
//		
//		Instances m_allInstances= getAllInsatnces(subClusters);
//		
//		m_classCentre= getClassCentre();
////		m_classCentre=getClassCentre(m_allInstances) ;
//		
//		m_globalBoundry= getGlobalBoundry();
////		m_globalBoundry= getGlobalBoundry(m_allInstances);
//		
////		m_globalAvDistance= getGlobalAvDistance();
//		m_globalAvDistance= getGlobalAvDistance(m_allInstances);
//		
//		m_classSD= getGlabalSD(m_allInstances);
//		
//		this.m_VDL= new double[this.NoOfSubClusters];
//		this.m_VDL= getVirtualDensityLoss();
//		this.m_maxVDL=getmaxVDL(this.m_VDL);
//		this.m_farthest_dis= getDistanceThreshold(this.m_centres, this.m_classCentre);
//		this.m_minGravF= getMinGravForce();
//		this.m_gravForce= new double[this.NoOfSubClusters][this.NoOfSubClusters];
//		this.m_gravForce= getGravForce();
//		subClusters.clear(); 
//		System.gc(); 
//		
//	}
	
//	private double[][] getGravForce() {
//		
//		double[][] GF= new double[this.NoOfSubClusters][this.NoOfSubClusters];
//		
//		for( int j=0; j< this.NoOfSubClusters; j++){
//			
//			for( int k=0; k< this.NoOfSubClusters; k++){
//				if( k!=j){
//					GF[j][k]= gravetationForce(this.m_size[k], this.m_size[j],
//					(Instance)this.m_centres[k].copy(),(Instance)this.m_centres[j].copy());
//				}
//		}
//		}
//return GF;
//	}
//
//	private double getMinGravForce() {
//	
//					double temp=0;
//					double minGF= Double.MAX_VALUE;
//					for( int j=0; j< this.NoOfSubClusters; j++){
//						for( int k=0; k< this.NoOfSubClusters; k++){
//							if( k!=j){
//						temp= gravetationForce(this.m_size[k], this.m_size[j],
//								(Instance)this.m_centres[k].copy(),(Instance)this.m_centres[j].copy());
//						if( temp< minGF)
//							minGF=temp;
//						}
//					}
//					}
//			return minGF;
//	}
//
//	private double gravetationForce(int sizei, int sizej, Instance m_centres2, Instance m_centres3) {
//		double distance;
////		  EuclideanDistance ED= new EuclideanDistance(this.m_centres);
//		  distance= CalculateED(m_centres2, m_centres3);
//		  double result= (double)(sizei*sizej)/ ( distance*distance);
//		 return result;
//	}
//
//	private double getmaxVDL(double[] loss) {
//		
//		double max=Double.MAX_VALUE;
//		for ( int i=0; i<loss.length;i++)
//			if(loss[i]<max )
//				max=loss[i];
//		return max;
//	}
//
//	private double getDistanceThreshold(Instance[] centres, Instance classCentre) {
//		//Get the farthest distance between any of the subclusters exist into the class
//		double furthestDistance=-1;
//		double temp=0;
//		for(int i=0; i< centres.length;i++ )
//		{
//					temp= CalculateED(centres[i], classCentre);
//					if( temp> furthestDistance) 
//						furthestDistance= temp;
//				
//			
//		}
//		
//		return furthestDistance;
//	}
//private double[] getVirtualDensityLoss() {
//	
//	double[] loss= new double[this.NoOfSubClusters];
//	double orgDensity=  densityFunction((double)this.m_totalsize, this.m_globalBoundry, 3);
//	for (int j = 0; j < this.NoOfSubClusters;  j++) {
//		int newsize= this.m_totalsize - this.m_size[j];	
//		double newRadius=getNewRadius(j);
//		double newDensity= densityFunction((double)newsize, newRadius,3) ;
//		loss[j]=newDensity-orgDensity  ; 
//				
//		}
//			return loss;
//			}
		
//		private double getNewRadius(int subClusterID) {
//			
//				Instance newCentre= getSubClustersCentre(subClusterID);
//				double boundry=0; 
//				double temp=0;
//				for ( int i=0; i< this.m_centres.length;i++){
//					if( i!=subClusterID){
//						temp= CalculateED(newCentre, this.m_centres[i])+ this.m_classBoundry[i];
//						if(temp> boundry) boundry=temp;
//					}
//				}
//				return boundry;
//			}


//		private Instance getSubClustersCentre(int exCluster) {
//			// Calculate Class centre with excluding sub-cluster
//			
//			Instance middle= (Instance) this.m_centres[0].copy();
//			double [] middleArray= new double [middle.numAttributes()];
//			int n= this.m_totalsize- this.m_size[exCluster];
//			
//			for( int i=0; i< middle.numAttributes();i++){
//				for( int j=0; j< this.NoOfSubClusters;j++)
//					if( j!=exCluster)
//						middleArray[i]+= this.m_centres[j].value(i)*this.m_size[j];
//				middleArray[i]/=( double) n;
//				
//			}
//			for( int i=0; i< middle.numAttributes();i++){
//			middle.setValue(i, middleArray[i]);
//			}
//			return middle;
//		}
//		
//
//		private double densityFunction(double size, double radius, int d) 
//		{ 
//			
//			if( size==0)
//				 return -1;
//			 double v= (4/3)* Math.PI*  Math.pow(radius, d);
//			 return size/v; 
////			if( size==0)
////			 return -1;
//////		 double v= (4/3)* Math.PI*  Math.pow(radius, d);
//////		 return size/v; 
////		 BigDecimal s =
////	            new BigDecimal(radius,new MathContext(20,RoundingMode.HALF_UP));
////		 s= s.pow(d,new MathContext(20,RoundingMode.HALF_UP));
////		BigDecimal PD= new BigDecimal(Math.PI);
////		BigDecimal f= new BigDecimal(4);
////		BigDecimal f1= new BigDecimal(3);
////		f=f.divide(f1, 50, RoundingMode.HALF_UP);
////		PD= PD.multiply(f,new MathContext(20,RoundingMode.HALF_UP));
////		
////		BigDecimal V=  s.multiply(PD,new MathContext(20,RoundingMode.HALF_UP));
////		BigDecimal m= new BigDecimal(size);
////		m=  m.divide(V, 20, RoundingMode.HALF_UP) ;
////		
////		double x= m.doubleValue();
////		return x;
//	}
//		
//	private double getGlobalAvDistance(Instances allInstances) {
//
//		Instance cntrInst= (Instance)this.m_classCentre.copy();
//		double distance=0;
//		for( int i=0; i<allInstances.size(); i++){
//			distance+= CalculateED(cntrInst, allInstances.get(i));
//			
//		}
//		return distance/this.m_totalsize;
//	}
////	private double getGlobalBoundry(Instances allInstances) {
////		double maxDistance=0;
////		Instance cntrInst= (Instance)this.m_classCentre.copy();
////		double distance=0;
////		for( int i=0; i<allInstances.size(); i++){
////			distance= CalculateED(cntrInst, allInstances.get(i));
////			if( distance> maxDistance)
////				maxDistance=distance;
////		}
////		return maxDistance;
////	}
//	private Instance getClassCentre(Instances allInstances) {
//		
//	
//		Instance middle= (Instance) this.m_centres[0].copy();
//		double [] middleArray= new double [middle.numAttributes()];
//		for( int i=0; i< middle.numAttributes();i++){
//				middleArray[i]=allInstances.meanOrMode(i);
//		}
//		for( int i=0; i< middle.numAttributes();i++){
//		middle.setValue(i, middleArray[i]);
//		}
//		return middle;
//	}
//	private Instances getAllInsatnces(ArrayList<Instances> subClusters) {
//		Instances allInstances= new Instances(removeClass(subClusters.get(0)));
//		for( int i=1; i<this.NoOfSubClusters;i++){
//			
//			Instances newInstances= new Instances(removeClass(subClusters.get(i)));
//			for( int j=0; j<newInstances.size();j++)
//				allInstances.add(newInstances.get(j));
//		} 
//
//		return allInstances;
//	}
//	
//	private Instance getGlabalSD(Instances allInstances) {
//	
//		double SD[]= new double[allInstances.numAttributes()];
//		Instance SDInstance= (Instance) allInstances.firstInstance().copy();
//		for( int i=0; i<allInstances.numAttributes();i++)
//				SD[i]=allInstances.variance(i);	
//		
//		for( int j=0; j< SD.length;j++){
//			SD[j]= Math.sqrt(SD[j]);
//			SDInstance.setValue(j ,SD[j]);
//		}
//		//inst.setClassValue(-1);
//			return SDInstance;
//	}
//	private double getGlobalAvDistance() {
//	
//		double Average=0; 
//		
//		for( int i=0; i< this.NoOfSubClusters;i++){
//			Average+=this.m_averageDistance[i]*this.m_size[i];
//		}
//		Average/= this.m_totalsize;
//		
//		return Average;
//	}
//	private double[] setAverageDistances(ArrayList<Instances> subClusters) {
//		double[] AvDistanceArr = new double[subClusters.size()];
//		for( int classID=0; classID< subClusters.size();classID++)
//		{
//		
//		Instances m_instances= new Instances(subClusters.get(classID));
//		Instance cntrInst= (Instance)this.m_centres[classID].copy();
//		double distance=0;
//		for( int i=0; i<m_instances.size(); i++){
//			distance+= CalculateED(cntrInst, m_instances.get(i));	
//		}
//		distance/= m_instances.size();
//		
//		AvDistanceArr[classID]=distance;
//		}
//		return AvDistanceArr;
//
//	}
//	public ClassWSubClusters(IntCWSCClass obj){
//		this.m_centres= new Instance[obj.m_centres.length];
//		this.m_size= new int[obj.m_size.length];
//		this.m_SD= new Instance[obj.m_SD.length];
//		this.m_classBoundry= new double[obj.m_classBoundry.length];
//		this.m_averageDistance= new double[obj.m_averageDistance.length];
//		this.m_VDL= new double[obj.m_VDL.length];
//		this.m_gravForce= new double[this.NoOfSubClusters][this.NoOfSubClusters];
//		this.label= obj.label; 
//		this.m_averageDistance= obj.m_averageDistance.clone(); 
//		this.m_centres= obj.m_centres.clone(); 
//		this.m_classBoundry= obj.m_classBoundry.clone(); 
//		this.m_classCentre= (Instance) obj.m_classCentre.copy(); 
//		this.m_classID= obj.m_classID; 
//		this.m_classSD= obj.m_classSD; 
//		this.m_farthest_dis= obj.m_farthest_dis; 
//		this.m_globalAvDistance= obj.m_globalAvDistance; 
//		this.m_globalBoundry= obj.m_globalBoundry; 
//		this.m_gravForce= obj.m_gravForce.clone(); 
//		this.m_maxVDL= obj.m_maxVDL; 
//		this.m_minGravF= obj.m_minGravF; 
//		this.m_SD= obj.m_SD.clone(); 
//		this.m_size= obj.m_size.clone(); 
//		this.m_totalsize= obj.m_totalsize; 
//		this.m_VDL= obj.m_VDL.clone(); 
//		this.NoOfSubClusters= obj.NoOfSubClusters; 
//	
//		
//	}
	public ClassWSubClusters(Instance[] m_centres,int [] m_size, Instance[] m_SD, double[] m_classBoundry, double[] m_averageDistance, 
		int NoOfSubClusters,String label, int classID, Instance classSD, double farthest_dis,double globalAvDistance, double globalBoundry, double[][] gravForce, 
		double maxVDL, double minGravF, int[] size, int totalsize, double[] VDL, Instance classCentre){
		
		this.m_centres= new Instance[m_centres.length];
		this.m_size= new int[m_size.length];
		this.m_SD= new Instance[m_SD.length];
		this.m_classBoundry= new double[m_classBoundry.length];
		this.m_averageDistance= new double[m_averageDistance.length];
		this.m_VDL= new double[VDL.length];
		this.m_gravForce= new double[NoOfSubClusters][NoOfSubClusters];
		this.label= label; 
		this.m_averageDistance= m_averageDistance.clone(); 
		this.m_centres= m_centres.clone(); 
		this.m_classBoundry= m_classBoundry.clone(); 
		this.m_classCentre= (Instance) classCentre.copy(); 
		this.m_classID= classID; 
		this.m_classSD= (Instance) classSD.copy(); 
		this.m_farthest_dis= farthest_dis; 
		this.m_globalAvDistance= globalAvDistance; 
		this.m_globalBoundry= globalBoundry; 
		this.m_gravForce= gravForce.clone(); 
		this.m_maxVDL= maxVDL; 
		this.m_minGravF= minGravF; 
		this.m_SD= m_SD; 
		this.m_size= size.clone(); 
		this.m_totalsize= totalsize; 
		this.m_VDL= VDL.clone(); 
		this.NoOfSubClusters= NoOfSubClusters; 
	
		
	}
	public ClassWSubClusters(ClassWSubClusters obj){
		this.m_centres= new Instance[obj.m_centres.length];
		this.m_size= new int[obj.m_size.length];
		this.m_SD= new Instance[obj.m_SD.length];
		this.m_classBoundry= new double[obj.m_classBoundry.length];
		this.m_averageDistance= new double[obj.m_averageDistance.length];
		this.m_VDL= new double[obj.m_VDL.length];
		this.m_gravForce= new double[this.NoOfSubClusters][this.NoOfSubClusters];
		
//		this.weights= new int[obj.weights.length];
		
		
		
	}
	
//	public ClassWSubClusters liteObject(){
//		ClassWSubClusters clearObj= new ClassWSubClusters (this); 
//		clearObj.label= this.label; 
//		clearObj.m_averageDistance= this.m_averageDistance.clone(); 
//		clearObj.m_centres= this.m_centres.clone(); 
//		clearObj.m_classBoundry= this.m_classBoundry.clone(); 
//		clearObj.m_classCentre= (Instance) this.m_classCentre.copy(); 
//		clearObj.m_classID= this.m_classID; 
//		clearObj.m_classSD= this.m_classSD; 
//		clearObj.m_farthest_dis= this.m_farthest_dis; 
//		clearObj.m_globalAvDistance= this.m_globalAvDistance; 
//		clearObj.m_globalBoundry= this.m_globalBoundry; 
//		clearObj.m_gravForce= this.m_gravForce.clone(); 
//		clearObj.m_maxVDL= this.m_maxVDL; 
//		clearObj.m_minGravF= this.m_minGravF; 
//		clearObj.m_SD= this.m_SD.clone(); 
//		clearObj.m_size= this.m_size.clone(); 
//		clearObj.m_totalsize= this.m_totalsize; 
//		clearObj.m_VDL= this.m_VDL.clone(); 
//		clearObj.NoOfSubClusters= this.NoOfSubClusters; 
//		return clearObj;
//		
//		
//	}
	
//	public ClassWSubClusters( int subClusters, Instance[] centres, Instance[] SD, double [] boundry,double[] averageD, int[] sizes){
//	this.NoOfSubClusters=subClusters;
//	this.m_centres= new Instance[subClusters];
//		this.m_centres= centres.clone(); 
//		this.m_size= new int[subClusters];
//		this.m_size=sizes.clone();
//		this.m_SD= new Instance[subClusters];
//		this.m_SD= SD.clone();
//		this.m_classBoundry=new double[subClusters];
//		this.m_classBoundry= boundry.clone();
//		this.m_averageDistance=new double[subClusters];
//		this.m_averageDistance= averageD.clone();
//		this.m_classCentre=getClassCentre() ;
//		this.m_globalBoundry= getGlobalBoundry();
//		this.m_globalAvDistance= getGlobalAvDistance();
////		Instances m_allInstances= getAllInsatnces(subClusters);
////		this.m_classSD=getGlabalSD(allInstances);
//	
//	}
	
//	private Instance getClassCentre() {
//		Instance middle= (Instance) this.m_centres[0].copy();
//		double [] middleArray= new double [middle.numAttributes()];
//		for( int i=0; i< middle.numAttributes();i++){
//			for( int j=0; j< this.NoOfSubClusters;j++)
//					middleArray[i]+= this.m_centres[j].value(i)*this.m_size[j];
//			middleArray[i]/=( double) this.m_totalsize;
//			
//		}
//		for( int i=0; i< middle.numAttributes();i++){
//			middle.setValue(i, middleArray[i]);
//		}
//		return middle;
//		
//	}	
//	private double getGlobalBoundry() {
//		
//		double temp=0; 
//		double furthest=0; 
//		
//		for( int i=0; i< this.NoOfSubClusters;i++){
//			temp= CalculateED(m_classCentre, m_centres[i]) + this.m_classBoundry[i];
//			if(temp> furthest)
//				furthest=temp;
//		}
//		return furthest;
//	}
//
//	private double[] setBoundries(ArrayList<Instances> subClusters) {
//		
//		double[] maxDistanceArr = new double[subClusters.size()];
//		
//		for( int classID=0; classID< subClusters.size();classID++)
//		{
//		
//		Instances m_instances= new Instances(subClusters.get(classID));
////		EuclideanDistance ED= new EuclideanDistance(m_instances);
////		ED.setDontNormalize(true);
//		double maxDistance=0;
//		Instance cntrInst= getClassCentre(m_instances);
//		double distance=0;
//		for( int i=0; i<m_instances.size(); i++){
//			distance= CalculateED(cntrInst, m_instances.get(i));
////			distance= ED.distance(cntrInst,  m_instances.get(i));
//		
//			if( distance> maxDistance)
//				maxDistance=distance;
//		}
//	
//		maxDistanceArr[classID]=maxDistance;
//		}
//		return maxDistanceArr;
//		
//	}
//
//	private Instance[] setSD(ArrayList<Instances> subClusters) {
//		Instance[] SDInstances= new Instance[this.NoOfSubClusters];
//		for( int classID=0; classID<this.NoOfSubClusters;classID++){
//			
//			
//			Instances m_instances= new Instances(removeClass(subClusters.get(classID)));
//			SDInstances[classID]= (Instance) m_instances.firstInstance().copy();
//			double SD[]= new double[m_instances.numAttributes()];
////			Instances microCluster= createMicroCluster(m_instances,classID,  3000);
//			for( int i=0; i<m_instances.numAttributes();i++)
//				SD[i]=m_instances.variance(i);
//		
//			for( int j=0; j< SD.length;j++){
//				SD[j]= Math.sqrt(SD[j]);
//				SDInstances[classID].setValue(j ,SD[j]);
//			}
//		
//		}
//		//inst.setClassValue(-1);
//			return SDInstances;
//	
//	}
//
//	private int[] setSizes(ArrayList<Instances> subClusters) {
//		
//		
//		int[] count= new int[ this.NoOfSubClusters];
//		for( int i=0; i< this.NoOfSubClusters;i++){
//		Instances m_instances= new Instances(subClusters.get(i));
//		 count[i]=m_instances.size();
//		}
//		return count;
//	}
//
//	private Instance[] setCentres(ArrayList<Instances> subClusters) {
//		
//		Instances new_Instances= new Instances(removeClass(subClusters.get(0)));
//		Instance[] centres= new Instance[this.NoOfSubClusters];
////		centresArr= new double[m_noClasses][centres.numAttributes()];
//		
//		 for( int classID=0; classID<this.NoOfSubClusters;classID++){
//		
//			 centres[classID]= (Instance)new_Instances.firstInstance().copy(); 
//			Instances m_instances= new Instances(removeClass(subClusters.get(classID)));
//			int m_numAtt= m_instances.numAttributes();
//			double[] mean= new double[m_numAtt];
//			 for( int i=0; i< m_numAtt;i++){
//				 mean[i]= m_instances.meanOrMode(i);
//			 }
//			
//			for( int i=0;i<m_numAtt;i++ )
//				centres[classID].setValue(i,mean[i]);
//		 }
//		
//		return centres;
//	
//	}

	public Instance [] getCentres(){
		return this.m_centres;
	
	}
	public int [] getSizes(){
		return this.m_size;
	
	}
	
	public Instance [] getSD(){
		return this.m_SD;
	
	}
	
	public double [] getBoundries(){
		return this.m_classBoundry;
	
	}
	public double [] getAvDistances(){
		return this.m_averageDistance;
	
	}
	public String getLable(){
		return this.label;
		
	}
	public int getID()
	{
		return this.m_classID;
	}
	


//private Instances removeClass(Instances inst) {
//    Remove af = new Remove();
//    Instances retI = null;
//    
//    try {
//      if (inst.classIndex() < 0) {
//	retI = inst;
//      } else {
//	af.setAttributeIndices(""+(inst.classIndex()+1));
//	af.setInvertSelection(false);
//	af.setInputFormat(inst);
//	retI = Filter.useFilter(inst, af);
//      }
//    } catch (Exception e) {
//      e.printStackTrace();
//    }
//    return retI;
//  }
//
//private double CalculateED(Instance first,Instance second ){
//	
//	
//	 double dist = 0.0;
//
//	    for (int i = 0; i < first.numAttributes(); i++) {
//	        double x = first.value(i);
//	        double y = second.value(i);
//
//	        if (Double.isNaN(x) || Double.isNaN(y)) {
//	            continue; // Mark missing attributes ('?') as NaN.
//	        }
//
//	        dist += (x-y)*(x-y);
//	    }
//
//	    return Math.sqrt(dist);
//	}

public double getCapacity() {
	// TODO Auto-generated method stub
	int capacity=0; 
	for( int i=0; i< m_size.length;i++)
		{
		capacity+= m_size[i];
		
		}
	return capacity;
}
public Object clone()

{

try

{

return super.clone();

}

catch ( CloneNotSupportedException e )

{

return null;

}

}

public ClassWSubClusters deepCopy() {
	// TODO Auto-generated method stub
	ClassWSubClusters obj = new ClassWSubClusters(this);
	obj.m_classID=this.m_classID;
	obj.label= this.label;
	obj.m_centres= this.m_centres.clone();
	obj.m_classBoundry= this.m_classBoundry.clone();
	obj.m_averageDistance=this.m_averageDistance.clone();
	obj.m_classCentre= (Instance) this.m_classCentre.copy();
	obj.m_SD= this.m_SD.clone();
	obj.m_size= this.m_size.clone();
	obj.m_totalsize= this.m_totalsize;
	obj.NoOfSubClusters= this.NoOfSubClusters;
	obj.m_globalBoundry= this.m_globalBoundry;
	obj.m_globalAvDistance=this.m_globalAvDistance;
	obj.m_classSD=(Instance) this.m_classSD.copy();
	obj.m_maxVDL= this.m_maxVDL;
	obj.m_VDL= this.m_VDL.clone();
	obj.m_farthest_dis= this.m_farthest_dis;
	obj.m_minGravF= this.m_minGravF;
//	obj.weights= this.weights.clone();
	obj.m_gravForce= this.m_gravForce.clone(); 
	return obj;
}

public int getNoSubClusters(){
	return NoOfSubClusters;
}

public void setInstances(Instances suspNovelInstances) {
	this.train_Instances= new Instances(suspNovelInstances);
	
}

public Instances getInstances() {
	// TODO Auto-generated method stub
	return this.train_Instances;
}
public double[][] getGravForce(){
	return this.m_gravForce; 
	
}
public double getGlobalAvDistance(){
	return this.m_globalAvDistance; 
}
public int  getClassID(){
	return this.m_classID; 
}
public double getFarthestdis(){
	return this.m_farthest_dis; 
}
public double getMaxVDL(){
	return this.m_maxVDL; 
}
public int getTotalsize(){
	return this.m_totalsize; 
}

public double getMinGravF(){
	return this.m_minGravF; 
}
public double[] getVDL(){
	return this.m_VDL; 
}
public Instance getClassSD(){
	return (Instance) this.m_classSD.copy(); 
}
public Instance getClassCentre(){
	return (Instance) this.m_classCentre.copy();
}
public double getClassBoundry() {
	// TODO Auto-generated method stub
	return this.m_globalBoundry; 
}

}




