/*************************************************************************
/* ECE 277: GPU Programmming 2022 FALL quarter
/* Author and Instructor: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

/*
Kwok Hung Ho
A15151703
*/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

float epsilon;
short *d_action;
short *d_astate; // agent state, 0: inactive, 1: active
float *rewards;

float alpha = 0.1f;
float gamma = 0.9f;
float deltae = 0.005f;
float *Q_Table;
static curandState *action_states = NULL;

// new variables for Q-lambda
#define COLS 32
#define ROWS 32
#define STATESPACE 1024
struct eligibilityTraces {
	int trace_length = 0; // length of trace
	int x[STATESPACE]; // history of state.x
	int y[STATESPACE]; // history of state.y
	int past_actions[STATESPACE]; // history of actions
	float E[STATESPACE]; // e-Trace value
};
eligibilityTraces *etrace;
float lambda=0.9;
bool *is_random;
int steps = 0;
int episode = 0;
int steps_per_episode[200];

// variables for measuring  kernel execution time
int total_steps = 0;
float runningsum = 0.0f;
float avg_kernel_time = 0.0f;

// Function to initialize random seed
__global__ void init_randsate ( curandState * state )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &state[tid]);
} 

// Function to initialize Q parallelly
__global__ void initQTable(float *Q_Table, int nx,int ny)	
{
	// unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ix = threadIdx.x;
    unsigned int iy = blockIdx.x;
    unsigned int idx = iy * nx + ix;
	if (ix < nx && iy < ny)
        Q_Table[idx] = 0;
}

// Function to initialize agents' states
__global__ void initStuff(short *d_astate, int nx)	
{
	unsigned int ix = threadIdx.x;
	if (ix < nx) {
		d_astate[ix] = 1; //active
	}
}
// Function to initialize agents' rewards
__global__ void initRewards(float* reward, int nx)
{
	unsigned int ix = threadIdx.x;
	if (ix < nx) {
		reward[ix] = 0;
	}
}
// Function to initialize agents' actions
__global__ void initActions(short* d_action, int nx)
{
	unsigned int ix = threadIdx.x;
	if (ix < nx) {
		d_action[ix] = 0;
	}
}

void agent_init()
{
	int size = 128 * sizeof(short);
	cudaMalloc((void**)&d_action, size); // action buffer
	cudaMalloc((void**)&d_astate, size);  // agent state
	cudaMalloc((void**)&rewards, size);  // rewards buffer
	initStuff <<<1,128>>> (d_astate,128); // init all agents to be active
	initRewards <<<1, 128 >>> (rewards, 128); // init rewards
	initActions <<<1, 128 >>> (d_action, 128); // init actions

	// init Q table
	int qsize = STATESPACE*4*sizeof(float); // 32x32 grid with 4 actions
	cudaMalloc((void **)&Q_Table, qsize);

	// init eligibility traces
	cudaMalloc((void **)&etrace, 128 * sizeof(eligibilityTraces)); // added for Q(lambda)

	// init is_random buffer
	cudaMalloc((void**)&is_random, 128*sizeof(bool));  // check is_random buffer

	// init Q Table
	initQTable <<< 4, STATESPACE >>>(Q_Table,STATESPACE,4);

	// random states init
	cudaMalloc((void **)&action_states, sizeof(curandState) * 128); // just 1 number (0,1,2,3), (0-1)
	init_randsate <<<1,128>>> (action_states); // seeding

	// init parameters
	epsilon=1.0;
}


__global__ void cuda_agent(curandState* actionState, int2* cstate, short* action, float* Q_table, float epsilon, short* d_astate, bool* is_random) 
{
	int tid = threadIdx.x;

	if (d_astate[tid] == 1 && tid<128) { // only take action if agent is active
		curandState *state = actionState + tid;
		float RANDOM = curand_uniform( state );

		is_random[tid] = false;

		// epsilon greedy
		if (RANDOM < epsilon) {
			float tmp = curand_uniform( state );
			action[tid] = min(3,short(tmp*4)); // pick random from [0,1,2,3]
			is_random[tid] = true;

		}
		else {
			int xc = cstate[tid].y*32 + cstate[tid].x; 
			short argmax = 0;
			float mx = Q_table[xc];
			for (short i = 0; i < 4; i++) {
				int idx = xc + i*STATESPACE; // argmax action for a given State
				if (Q_table[idx]>mx) {
					mx = Q_table[idx];
					argmax = i;
				}
			}
			action[tid] = argmax;
		}
	}
}

void agent_init_episode() 
{
	initStuff <<<1,128>>> (d_astate,128); // set all 128 agents active
	initRewards <<<1, 128 >>> (rewards, 128);
	initActions <<<1, 128 >>> (d_action, 128);
}

float agent_adjustepsilon() 
{
	if (episode < (1/deltae))
		steps_per_episode[episode] = steps;
	episode++;
	//printf("episode %d took %d steps\n", episode, steps);
	steps = 0;
	epsilon = epsilon - deltae; 
	return epsilon; // return cpu variable (epsilon)
}

short* agent_action(int2* cstate)
{
	steps++;
	cuda_agent <<<1, 128 >>>(action_states, cstate, d_action, Q_Table, epsilon, d_astate, is_random);
	return d_action;
}

// Function to update Q and set agents inactive based on reward
__global__ void updateQLam(int2* cstate, int2* nstate, float *rewards, short* action, float* Q, float alpha, float gamma, short* d_astate, float lambda, eligibilityTraces *etrace, short* d_action, bool* is_random) 
{
	unsigned int tid = threadIdx.x;
	if (tid < 128) {

		if (d_astate[tid]==1) { // only update if agent is active

			// action is y, state is x
			int y_ = action[tid];
			int xc = cstate[tid].y * 32 + cstate[tid].x;
			int xn = nstate[tid].y * 32 + nstate[tid].x;
			int curi = y_ * (STATESPACE)+xc;

			// getting argmax & max
			float mx = Q[xn];
			short argmax = 0;
#pragma unroll
			for (short i = 0; i < 4; i++) {
				int nidx = xn + i * (STATESPACE); // argmax action for a given next State
				if (Q[nidx] > mx) {
					mx = Q[nidx];
					argmax = i;
				}
			}

			// update traces
			int idx = (int)etrace[tid].trace_length; // getting current step
			etrace[tid].past_actions[idx] = d_action[tid]; // setting previous action
			etrace[tid].x[idx] = cstate[tid].x; // setting previous state
			etrace[tid].y[idx] = cstate[tid].y; // setting previous state
			etrace[tid].trace_length = idx + 1; // incrementing step

			// process reward
			//if (rewards[tid] == 1) {  // catch mine
			if (rewards[tid] != 0) {  // catch mine or flag
				//Q[curi] = Q[curi] + alpha * (rewards[tid] - Q[curi]);
				float delta = rewards[tid] - Q[curi];
				// get current step
				int curstep = etrace[tid].trace_length;
				// set prev step to be valid
				etrace[tid].E[curstep-1] = 1; // replacing trace
#pragma unroll
				for (int i = curstep - 1; i > 0; --i) {
					int x = etrace[tid].x[i];
					int y = etrace[tid].y[i];
					int qid = etrace[tid].past_actions[i]*ROWS*COLS + (y*COLS + x);
					// set Q for all old traces
					Q[qid] += alpha * delta * etrace[tid].E[i];
					// update E, might need israndom
					etrace[tid].E[i-1] = gamma * lambda * etrace[tid].E[i];
				}
				d_astate[tid] = 0; // disable agent
				
			}
			/*else if (rewards[tid] == -1) {
				Q[curi] = Q[curi] + alpha * (rewards[tid] - Q[curi]);
			}*/
			else { // neither flag nor mine
				Q[curi] = Q[curi] + alpha * (rewards[tid] + gamma * mx - Q[curi]);
			}

			//if (is_random[tid] == true) {
			if (argmax != d_action[tid]) {
				int n = etrace[tid].trace_length;
				etrace[tid].trace_length=0; // reset trace_length

#pragma unroll
				for (int i = n; i >= 0; --i) {
					etrace[tid].E[i]=0;
				}
			}
		}
	}
}

__global__ void updateQ(int2* cstate, int2* nstate, float* rewards, short* action, float* Q, float alpha, float gamma, short* d_astate, float lambda, eligibilityTraces* etrace, short* d_action, bool* is_random)
{
	unsigned int tid = threadIdx.x;
	if (tid < 128) {

		if (d_astate[tid] == 1) { // only update if agent is active

			// action is y, state is x
			int y_ = action[tid];
			int xc = cstate[tid].y * 32 + cstate[tid].x;
			int xn = nstate[tid].y * 32 + nstate[tid].x;
			int curi = y_ * (STATESPACE)+xc;

			// getting argmax & max
			float mx = Q[xn];
			short argmax = 0;
#pragma unroll
			for (short i = 0; i < 4; i++) {
				int nidx = xn + i * (STATESPACE); // argmax action for a given next State
				if (Q[nidx] > mx) {
					mx = Q[nidx];
					argmax = i;
				}
			}

			if (rewards[tid] != 0) {  // catch mine or flag
				Q[curi] = Q[curi] + alpha * (rewards[tid] - Q[curi]);
				d_astate[tid] = 0; // disable agent

			}
			else { // neither flag nor mine
				Q[curi] = Q[curi] + alpha * (rewards[tid] + gamma * mx - Q[curi]);
			}
		}
	}
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// update Q table for all 128 agents
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	updateQLam <<< 1, 128 >>>(cstate, nstate, rewards, d_action, Q_Table, alpha, gamma, d_astate, lambda, etrace, d_action, is_random);
	//updateQ << < 1, 128 >> > (cstate, nstate, rewards, d_action, Q_Table, alpha, gamma, d_astate, lambda, etrace, d_action, is_random);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, stop);
	
	runningsum += kernel_time;
	total_steps++;
	avg_kernel_time = runningsum / total_steps;
	//printf("kernel exec time \t\t\t: %f ms\n", kernel_time);
	//printf("kernel average time \t\t\t: %f ms\n", avg_kernel_time);
	
}
