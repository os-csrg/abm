#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h> 
#include <sys/stat.h>
#include <mpi.h>
#include <ctime>
#include <algorithm> 
using namespace std;

#include "CImg.h" // has to be in the same directory for compilation
using namespace cimg_library;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// ========================================================================
// Useful Snippets:
// mpic++ -o abm abm.cpp
// mpirun -np 4 abm
// ========================================================================

// ========================================================================
// Constant variable definitions (can be parameterised)
// ========================================================================
const int worldWidth = 500;
const int worldHeight = 500; // should be the same (different = untested)
const int SEND_BUFFER_SIZE = worldWidth * worldHeight;
const int numTrades = 100000000; // trades count 10000
const int agentStartingWealth = 100;
// model specific variables, check method 'slave' for implementation
double SchellingSatisfaction = .00;
double fixedFraction = .1;

// ========================================================================
// Class definitions (can be redacted for performance)
// ========================================================================

class Agent{
	//public = faster, could be in struct to squeese even more cycles
	public:
		int balance;
		int numNbrs; // number of neighbours (precalculated for speed)
		Agent* neighbours[4]; // pointers to the neighbouring agents
		int fixedSize = 0;
		// operator for sorting using the standard library
	 	bool operator < (const Agent &rhs) const { return (balance < rhs.balance); }
	};

class World{ // no class = faster, hold plain 2D array for max performance
	// holds all the agents [will be expanded for functionality]
	public:
		Agent* agents = new Agent[worldWidth * worldHeight];
	};

// ========================================================================
// OpenMPI setup and configuration (shouldn't need changing ever)
// ========================================================================

void master(int numprocs, int rank);
void slave(int numprocs, int rank);
int main(int argc, char** argv) {
	// general timing features
	std::clock_t start;
	start = std::clock();

	string path = SchellingSatisfaction + "_" + fixedFraction;
	mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// Get the rank of current process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	// if this is the master process, collect data from other processes as they finish
	//if(world_rank == 0)	master(world_size, world_rank);
	// else perform computation and save result
	//else 
	slave(world_size, world_rank);	
	// Finalize the MPI environment. No more MPI calls can be made after this
	MPI_Finalize();
	// end of timing things
	double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "Execution Duration: " << duration << endl;
	exit(0);
	}

// ========================================================================
// model functions including trade rules and pairing rules below
// ========================================================================

inline void setNeighbours(World* w, int x, int y){
	// NOTE: can be tweaked so only x&y are given
	// adds references to VN neighbours (where radius = 1)
	int upX = x;
	int upY = y-1;
	if(upY < 0) upY = worldHeight - 1;

	int downX = x;
	int downY = y + 1;
	if(downY > worldHeight - 1) downY = 0;

	int leftX = x - 1;
	int leftY = y;
	if(leftX < 0) leftX = worldWidth - 1; 

	int rightX = x + 1;
	int rightY = y;
	if(rightX > worldWidth - 1) rightX = 0;

	w->agents[worldWidth * y + x].neighbours[0] = &w->agents[worldWidth * upY + upX]; // up
	w->agents[worldWidth * y + x].neighbours[1] = &w->agents[worldWidth * downY + downX]; // down
	w->agents[worldWidth * y + x].neighbours[2] = &w->agents[worldWidth * leftY + leftX]; // left
	w->agents[worldWidth * y + x].neighbours[3] = &w->agents[worldWidth * rightY + rightX]; // right
	w->agents[worldWidth * y + x].numNbrs = 4;

	// debug output
	// cout << "[" << upX << "," << upY << "],"
	// 	 << "[" << rightX << "," << rightY << "],"
	// 	 << "[" << downX << "," << downY << "],"
	// 	 << "[" << leftX << "," << leftY << "]" << endl;
	
	}

inline void tradeFixedAmount(Agent* winner, Agent* loser, int amount){
	winner->balance += amount;
	loser->balance -= amount;
	}

inline void tradeRandFraction(Agent* winner, Agent* loser){
	// trades random fraction of poorest agent's wealth
	int amount;
	if(winner->balance > loser->balance) amount = rand() % loser->balance;
	else amount = rand() % winner->balance;
	winner->balance += amount;
	loser->balance -= amount;
	}	

inline void tradeFixedFraction(Agent* winner, Agent* loser){
	int amount = fixedFraction * loser->balance;
	winner->balance += amount;
	loser->balance -= amount;
	}

inline void swapPlaces(Agent& a, Agent& b){
	double balance = a.balance;
	a.balance = b.balance;
	b.balance = balance;
	}

inline bool isSatisfied(Agent* agent, double satisfaction){
	double neighboursBalance = 0;
	for(int i = 0; i < agent->numNbrs; i++){ // cumulate the neighbours
		neighboursBalance += agent->neighbours[i]->balance;
	}
	
	double aveNeighbourBalance = neighboursBalance / agent->numNbrs;
	double satisfactionAmount = agent->balance * satisfaction;

	if(aveNeighbourBalance < agent->balance - satisfactionAmount || 
	   aveNeighbourBalance > agent->balance + satisfactionAmount)
		return false;	
	return true;
	}


inline double CalcSegregationFactor(World world)
{
	double overallValue = 0;
	double addFraction = .25;
	for (int f = 0; f < worldWidth * worldHeight; f++)
	{
		Agent *agent = &world.agents[f];
		for (int n = 0; n < 4; n++)
			if (agent->neighbours[n]->fixedSize == agent->fixedSize)
				overallValue += addFraction;

	}
	return overallValue/(worldWidth * worldHeight);


}	
const int numBounds = 3;
//double segregationValues[numTrades / 10000];
int segregationValuesArrayIndex = 0;
int timeStepIndex = 0;
void CalcSegMidRun(World world, int timeStep,int rank)
{
	double minBalance = agentStartingWealth;
	double maxBalance = 0;
	// get min and max values of balances
	for (int y = 0; y < worldWidth; y++)
	{
		for (int x = 0; x < worldHeight; x++)
		{
			double balance = world.agents[worldWidth * y + x].balance;

			if (balance > maxBalance) maxBalance = balance;
			if (balance < minBalance) minBalance = balance;
		}
	}


	double bounds[numBounds]{};

	// find the values that the number of bounds fit to
	for (size_t i = 0; i < numBounds; i++)
	{
		bounds[i] = (maxBalance - minBalance) / (numBounds) * (i + 1);
	}

	// for each agent, allocate a bound value
	for (int y = 0; y < worldWidth; y++)
	{
		for (int x = 0; x < worldHeight; x++)
		{
			Agent *agent = &world.agents[worldWidth * y + x];
			double money = agent->balance;
			int boundColour = 0;

			// clamp infinite values to numBounds variable
			for (size_t i = 0; i < numBounds; i++)
			{
				if (money <= bounds[i])
				{
					boundColour = i;
					agent->fixedSize = i;

					break;
				}
			}
		}
	}

	double segVal = CalcSegregationFactor(world);
	//segregationValues[segregationValuesArrayIndex++] = segVal;

	std::string str = std::to_string(SchellingSatisfaction);

	ofstream myfile;

	string path = SchellingSatisfaction + "_" + fixedFraction;

	string file = "output/" + to_string(worldWidth) + "/";
	//string file = path + "/";
	myfile.open(file + to_string(rank)  + ".csv", std::ios::app);

	//myfile << segVal << " , " << "\r\n";
	double nroamlisedVal = (double)timeStepIndex / 1421;
	myfile << nroamlisedVal << "," << segVal << "\r\n";

	timeStepIndex++;
	myfile.close();


}
inline double scale(double valueIn, double baseMin, double baseMax, double limitMin, double limitMax);
void CimgSave(World world, int rank){
	
	CImg<float> image(worldWidth, worldHeight, 1, 3, 0);
	double minBalance = agentStartingWealth;
	double maxBalance = 0;
	for (int y = 0; y < worldWidth; y++)
	{
		for (int x = 0; x < worldHeight; x++)
		{
			double balance = world.agents[worldWidth * y + x].balance;
			
			if(balance > maxBalance) maxBalance = balance;
			if(balance < minBalance) minBalance = balance;
		}
	}
	
	double bounds[numBounds]{};

	//assumming 0 is min
	for (size_t i = 0; i < numBounds; i++)
	{
		bounds[i] = maxBalance / (numBounds) * (i + 1);

	}
	

	for (int y = 0; y < worldWidth; y++)
	{
		for (int x = 0; x < worldHeight; x++)
		{
			Agent *agent = &world.agents[worldWidth * y + x];
			double money = agent->balance;
			int boundColour = 0;

			for (size_t i = 0; i < numBounds; i++)
			{
				if (money <= bounds[i])
				{
					boundColour = i;
					agent->fixedSize = i;

					break;
				}
			}

			//double scaledMoney = scale(money, minBalance, maxBalance, 0, 1);

			//double pixel = 255 * scaledMoney;

			double pixel = 255 / (numBounds - 1)*(boundColour);
			float color[] = { pixel, 0, 0 };
			image.draw_point(x, y, color);
		}
	}
	
	double segVal = CalcSegregationFactor(world);
	//change between incrementing schelling or trading fractions
	std::string str = std::to_string (SchellingSatisfaction);
	//std::string str = std::to_string (fixedFraction);
	
	
	//remove leading zeros from decimal_point
	str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
	
	string file = "output/" + to_string(worldWidth) + "/"+to_string(rank) + " " + to_string(segVal) + " " + str + ".bmp";	
	

	
	image.save(file.c_str());
	}

// some maths funtions which may be useful

inline double scale(double valueIn, double baseMin, double baseMax, double limitMin, double limitMax) {
	return ((limitMax - limitMin) * (valueIn - baseMin) / (baseMax - baseMin)) + limitMin;
	}

inline void SchellingStyleModel(World world, int rank){
	// schelling model
	// pick a random agent
	int winX = rand()%worldWidth;
	int winY = rand()%worldHeight;
	Agent *winner = &world.agents[worldWidth * winY + winX];
	Agent *loser;		
	if(isSatisfied(winner, SchellingSatisfaction)){
		// agent is satisfied with wealth of its neighbours
		loser = winner->neighbours[rand()%winner->numNbrs];
		tradeFixedFraction(winner, loser);
	}else{ // agent is not satisfied, should swap with another unsatisfied agent
		for(int f = 0; f < worldWidth * worldHeight; f++){
			int randX = rand()%worldWidth;
			int randY = rand()%worldHeight;
			Agent *randomAgent = &world.agents[worldWidth * randY + randX];
			if(randomAgent == winner) continue;
			if(!isSatisfied(randomAgent, SchellingSatisfaction)){
				swapPlaces(*winner, *randomAgent);
				break;
			}
		}
	}
}


void slave(int numprocs, int rank){
	// seed the rand with the rank (could be better)
	srand(1000);
	// scale a variable using the rank of the process

	
	//change between incrementing schelling or trading fractions
	SchellingSatisfaction += (.01 * (rank-1));
	//fixedFraction = (.001 * (rank-1));
	
	// create world to use
	World world;
	// initialise the agents with a set amount of money
	for(int y = 0; y < worldWidth; y++){
		for(int x = 0; x < worldHeight; x++){
			world.agents[worldWidth * y + x].balance = rand()%agentStartingWealth;
			// add neighbouring links for each agent
			setNeighbours(&world, x, y);
		}
	}
	int numUnsatisfiedAgents = 0;
	for(int y = 0; y < worldWidth; y++){
		for(int x = 0; x < worldHeight; x++){
			if(!isSatisfied(&world.agents[worldWidth * y + x], SchellingSatisfaction)){
				numUnsatisfiedAgents ++;
			}			
		}
	}
	cout << "numUnsatisfiedAgents: " << numUnsatisfiedAgents << endl;
	// perform the trades (includes selection process for now)
	double modValue = 1;
	for(int t = 0; t < numTrades; t++){

		if ((t % (int)modValue) == 0)
		{
			modValue *= 1.01;
			CalcSegMidRun(world, t, rank);
		}


		SchellingStyleModel(world, rank);

		// ========================================================
		// Trading selection (also multiple types)
		// ========================================================
		//tradeFixedAmount(winner, loser, 10);
		//tradeRandFraction(winner, loser);
		//tradeFixedFraction(winner, loser);
	}


	//for (int i = 0; i < numTrades / 10000; i++)
	//{
	//	std::string str = std::to_string(SchellingSatisfaction);

	//	ofstream myfile;
	//	string file = "output/" + to_string(worldWidth) + "/";
	//	myfile.open(file + to_string(rank) + " " + str + ".txt", std::ios::app);

	//	myfile << segVal << endl;
	//	myfile.close();

	//}
	// save this thread's output as an image
	//CimgSave(world, rank);
	

	delete [] world.agents;	

	}

// ========================================================================
// Master process, result gathering and analysis done here
// ========================================================================

void master(int numprocs, int rank){
	

	} 