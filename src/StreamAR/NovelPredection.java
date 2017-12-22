package StreamAR;

import weka.core.Instances;

public class NovelPredection {
//	protected Instances m_novelInstances;
//	protected Instances m_recInstances;
	
	private boolean m_faNovel= false;
	private boolean m_recurrent= false;
	private boolean m_unknown=false; 
	private String m_str;
	private  int m_classSize=0; 
	private String[] classLabel;
	private int novelNumber=0;
//	private static double m_Fp, m_Fn,m_Tp, m_Tn;

	// Distance>> Density>> Grav>> SD
	private  String[] m_measNoveltyType;
	private  String m_noveltyType="Unknown";
	private double n_novel;
	private String m_predLabel;
	private boolean densityF= false ; 
	private boolean gravityF= false; 
// Measure array has one value 
	public NovelPredection( boolean b, boolean recurrent, String s){
	m_faNovel=b;
	m_recurrent= recurrent; 
	m_str=s;
//	m_novelInstances= new Instances(novel);
//	m_recInstances= new Instances( exist);
	}
	
	public void setNovelNumber(int n){
		novelNumber= n;
		
	}
	public void setDenNovFlag(boolean b){
		this.densityF=b;
		
	}
	public boolean getDenNovFlag()
	{
		return densityF; 
	}
	
	public void setGravNovFlag(boolean b){
		this.gravityF=b;
		
	}
	public boolean getGravNovFlag()
	{
		return gravityF; 
	}
	public int getNovelNumber(){
		return novelNumber;
		
	}
	
	public void setNoveltyType(String s){
		this.m_noveltyType= s.trim();
		
	}

	public String getNoveltyType(){
		return this.m_noveltyType.trim();
	}

	public void setm_measNoveltyType(String[] s){
		this.m_measNoveltyType= s.clone();
		
	}

	public String[] getm_measNoveltyType(){
		// Distance>> Density>> Grav>> SD
		return this.m_measNoveltyType;
	}

//	public void setFp(double Fp ){
//	m_Fp= Fp;
//	}
//
//	public double getFp( ){
//	 return m_Fp;
//	 }
//
//	public void setFn(double Fn ){
//		m_Fn= Fn;
//	}
//
//	public double getFn( ){
//		 return m_Fn;
//	}
//	
//	public void setTp(double Tp ){
//		m_Tp= Tp;
//	}
//	
//	public double getTp( ){
//		return m_Tp;
//	}
//		
//	public double getTn( ){
//		return m_Tn;
//	 }
//	public void setTn(double Tn ){
//		m_Tn= Tn;
//		}
public boolean isUnknown(){
	return this.m_unknown; 
}
public boolean IsFAnovel() {
	// TODO Auto-generated method stub
	return m_faNovel;
}
public String getDetails() {
	// TODO Auto-generated method stub
	
	return m_str;
}
public void setUnknown(){
	this.m_unknown=true; 
}
public void setClassTrueLabel(String[] s){
	classLabel=s;
}
public void setClassSize(int i){
	m_classSize= i;
}

public void setNoofNovelInstances(int i){
	 n_novel = i;
}
public double getNoofNovelInsatces(){
	return n_novel; 
}

public String[] getClassTrueLabel(){
	return classLabel;
}
public int getClassSize(){
	return m_classSize;
}
//public void addNovelInstance(Instance novel){
//	this.m_novelInstances.add(novel);
//}
//public Instances getNovelInstances(){
//	return m_novelInstances;
//}
//
//public Instances

public void setPredictedLabel(String PL) {
	this.m_predLabel= PL;
	
}
public boolean isRecurrent(){
	return this.m_recurrent; 
}

public String getPredictedLabel(){
	return this.m_predLabel; 
}

}
