package StreamAR;



public class Prediction {
	
	protected int segID;
	protected String[] actLabel;
	protected double purity=0; 
	
	protected String type; 
	protected String label;
	protected String trueLabel; 
	 protected String str;
	protected boolean[] measures; 
	protected double confLevel;
	protected int segSize; 
	// List of instances
	//protected ArrayList<Instances> m_model= new ArrayList<>();
	
	public void setParameters(int segNo, String pLabel, String cLabel){
		// other predictions 		
		segID= segNo;
		label= pLabel; 
		trueLabel= cLabel;
		measures= new boolean[4];
		confLevel=0;
		if(pLabel.contains("Unrecognised")) type="Unrecognised"; 
}
	public void setActiveParameters(int segNo, String[] pLabel, String cLabel){
		// Active prediction 		
		segID= segNo;
		actLabel= pLabel; 
		trueLabel= cLabel;
		measures= new boolean[4];
		confLevel= 0.5;
		type= "Active";
}
	
	public Prediction() {
		// TODO Auto-generated constructor stub
	}
	
	public void setSize( int s){
		segSize=s;
		
	}
	public int  getSize(){
		 return segSize;
		
	}
	public void setMeasures(boolean density, boolean distance, boolean SD, boolean gravity)
	{
		measures[0]= density; 
		measures[1]= distance;
		measures[2]= SD;
		measures[3]= gravity; 
		
		//not active or unrecognised
		if( type==null)
			setConfidencelevel();
	}
	public void setMeasures(boolean density, boolean distance,boolean gravity)
	{
		measures[0]= density; 
		measures[1]= distance;
		measures[2]= gravity;
		//not active or unrecognised
		if( type==null)
			setConfidencelevel_3();
	}
		private void setConfidencelevel_3() {
			int correct=0; 
			int incorrect=0; 
			for( int i=0; i< 3; i++){
				if(measures[i])
					correct++;
			}
			incorrect= 3- correct;
			if(incorrect> correct)
				{
				confLevel= incorrect*0.25;
				type="Incorrect";
				}
			else
				{
				confLevel= correct* 0.25;
				type="Correct";
				}
			
		
	}
		private void setConfidencelevel() {
		// TODO Auto-generated method stub
			int correct=0; 
			int incorrect=0; 
			for( int i=0; i< 4; i++){
				if(measures[i])
					correct++;
			}
			incorrect= 4- correct;
			if(incorrect> correct)
				{
				confLevel= incorrect*0.25;
				type="Incorrect";
				}
			else
				{
				confLevel= correct* 0.25;
				type="Correct";
				}
			
	}
		public String getTrueLabel() {
			// TODO Auto-generated method stub
			return trueLabel;
		}
		public String getPredLabel() {
			// TODO Auto-generated method stub
			String s;
			if( type== "Active")
				s= actLabel[0]+" "+ actLabel[1];
			else
				s= label;
			return s;
		}
		
		public String getType() {
			// TODO Auto-generated method stub
			return type;
		}
		public String getConfLevel() {
			// TODO Auto-generated method stub
			return Double.toString(confLevel);
		}
		public void setStatatsistics( String s)
		{
			str= s;
		}
		public String segStatistics(){
			return str;	
		}
		public boolean[] getMeasures() {
			// TODO Auto-generated method stub
			return measures;
		}
		public void setPurity( double m){
			purity= m; 
			
		}
		public double getPurity( ){
			return purity; 
			
		}
			
}
