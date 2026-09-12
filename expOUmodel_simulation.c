#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// compile command: gcc expOUmodel_simulation.c -lm -O3

#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI

#define N_BIN 129
#define DIM 3

#define NSKIP 1e7 // the initial number of steps to determine the bounding box
#define NSTEPS 1e10  // the length of the stochastic trajectory that we bin
#define TIME_STEP 1e-3
#define INFTY 1.0e6

//--- model parameters
#define R_PAR 0.4
#define A_PAR 0.01
#define K_PAR 0.3
#define B_PAR 0.4
#define D_PAR 0.4
#define V1_PAR 0.2
#define V2_PAR 0.2
#define S_PAR 0.2
#define GMIN 0.15 // GMIN = S_PAR*K_PAR/B_PAR
#define DELTA 0.2885 // g1 = 0.4385, gmin = 0.15, delta = g1 - gmin

//--- Ornstein-Uhlenbeck parameters
#define ALPHA 0.1 //0.1329 // alpha = 0.1329 is the resonance frequency
#define SIGMA 0.1  


//----- main	
int main(void);
void box_mueller(double *w,int nw); // generates a pair of Gaussian random variables N(0,1)
void OUstep(double *u, double wval, double dt, double sqrt_dt);
void model(double *f, double *pop, double uval);
void RK4step(double *f, double *pop, double uval, double dt);
void initial_run(int Nsteps, double *f, double *pop, double *pp_min, 
		double *pp_max, double *umin, double *umax);
void binning_trajectory(long Nsteps,long *bins,long *ubins,double *f,double *pop,
			double *pp_min,double *pp_step,double *umin,double *ustep);



//------------------------------

void box_mueller(double *w,int nw){
	// generates nw Gaussian random variables
	double x1, y1, p, q;
	int i, j;
	j = (nw%2 == 0) ? nw/2 : nw/2 + 1;
	for( i = 0; i < j; i++ ) {
		do{
			p=random();
			x1 = p/RAND_MAX;
			p=random();
			y1 = p/RAND_MAX;
		}
		while( x1 == 0.0 );
		/* Box-Muller transform */
		p=PI2*y1;
		q=2.0*log(x1);
		w[i]=cos(p)*sqrt(-q);
		if( i + j < nw ) w[i + j]=sin(p)*sqrt(-q);
	}	
}

void OUstep(double *u, double wval, double dt, double sqrt_dt) {
// 	printf("u = %.4e, dt = %.4e, sqrt_dt = %.4e, wval = %.4e\n",*u,dt,sqrt_dt,wval);
	(*u) += -ALPHA*(*u)*dt + SIGMA*sqrt_dt*wval;
// 	printf("u = %.4e, dt = %.4e, sqrt_dt = %.4e, wval = %.4e\n",*u,dt,sqrt_dt,wval);
}


void model(double *f, double *pop, double uval) {
	double g = GMIN + DELTA*exp(uval);
	
	f[0] = pop[0]*(R_PAR - A_PAR*pop[0] + S_PAR*pop[1] - B_PAR*pop[2]);
	f[1] = K_PAR*pop[0]*pop[2] - pop[1]*(g*pop[0] + D_PAR + V1_PAR);
	f[2] = D_PAR*pop[1] - V2_PAR*pop[2];

}

void RK4step(double *f, double *pop, double uval, double dt) {
	double k1[DIM],k2[DIM],k3[DIM],k4[DIM];
	double *arg;
	int j;
	
	arg = (double *)malloc(DIM*sizeof(double));
	// stage k1
	model(f,pop,uval);
	for( j = 0; j < DIM; j++) {
		k1[j] = f[j];
		arg[j] = pop[j] + 0.5*dt*k1[j];
	}
	// stage k2
	model(f,arg,uval);
	for( j = 0; j < DIM; j++) {
		k2[j] = f[j];
		arg[j] = pop[j] + 0.5*dt*k2[j];
	}
	// stage k3
	model(f,arg,uval);
	for( j = 0; j < DIM; j++) {
		k3[j] = f[j];
		arg[j] = pop[j] + dt*k3[j];
	}
	// stage k4
	model(f,arg,uval);
	for( j = 0; j < DIM; j++) {
		k4[j] = f[j];
		pop[j] += dt*(k1[j] + 2.0*(k2[j] + k3[j]) + k4[j])/6.0;
	}

}

void initial_run(int Nsteps, double *f, double *pop, double *pp_min, double *pp_max, double *umin, double *umax) {
	int j;
	double *w, *u;
	double dt = TIME_STEP, sqrt_dt = sqrt(TIME_STEP);
	
	w = (double *)malloc(Nsteps*sizeof(double));
	u = (double *)malloc(sizeof(double));
	*u = 0.0;

	box_mueller(w,Nsteps);
	for( j = 0; j < Nsteps; j++ ) {
		OUstep(u,w[j],dt,sqrt_dt);
// 		printf("u = %.4e, dt = %.4e, sqrt_dt = %.4e, wval = %.4e\n",*u,dt,sqrt_dt,w[j]);
		
		*umin = min(*umin,*u);
		*umax = max(*umax,*u);
		
		RK4step(f,pop,*u,dt);
// 		printf("u = %.4e, dt = %.4e, sqrt_dt = %.4e, wval = %.4e\n",*u,dt,sqrt_dt,w[j]);
		
		pp_min[0] = min(pp_min[0],pop[0]);
		pp_max[0] = max(pp_max[0],pop[0]);
		pp_min[1] = min(pp_min[1],pop[1] + pop[2]);
		pp_max[1] = max(pp_max[1],pop[1] + pop[2]);
		
	}
	free(w);
	free(u);
	
}

void binning_trajectory(long Nsteps,long *bins,long *ubins,double *f,double *pop,
			double *pp_min,double *pp_step,double *umin,double *ustep) {
	int j, j1, j2, ind,  Nw = 10000;
	long jstep = 0;
	int jrep;
	long jprint = Nsteps/10;
	double *w, *u;
	double dt = TIME_STEP, sqrt_dt = sqrt(TIME_STEP);
	
	w = (double *)malloc(Nw*sizeof(double));
	u = (double *)malloc(sizeof(double));
	*u = 0.0;
	
	while( jstep < Nsteps ) {
		box_mueller(w,Nw);
		for( j = 0; j < Nw; j++ ) {
			OUstep(u,w[j],dt,sqrt_dt);
// 			printf("u = %.4e, dt = %.4e, sqrt_dt = %.4e, wval = %.4e\n",*u,dt,sqrt_dt,w[j]);

			RK4step(f,pop,*u,dt);
			j1 = min(max(0,(int)floor((pop[0] - pp_min[0])/pp_step[0]+0.5)),N_BIN-1);
			j2 = min(max(0,(int)floor((pop[1] + pop[2] - pp_min[1])/pp_step[1] +0.5)),N_BIN-1);
			ind = j1 + j2*N_BIN;
			bins[ind]++;
			j1 = min(max(0,(int)floor(((*u) - (*umin))/(*ustep)+0.5)),N_BIN-1);
// 			printf("j1 = %i, aux = %i\n",j1,(int)floor(((*u) - (*umin))/(*ustep)+0.5));
			ubins[j1]++;
			jstep++;
		}
		if( jstep%jprint == 0 ) {
			printf("jstep = %li\n",jstep);				
		}
	}
	free(w);
	free(u);	
}




//------------- M A I N
int main() {
	double *f, *pop, *pp_min, *pp_max, *umin, *umax;
	double *pp_step, *ustep;
	double aux;
	long *bins;
	long *ubins;
	double gbar = GMIN + DELTA;
	int Npp = DIM-1, j, Nbins = N_BIN*N_BIN, ind;
	clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time

	
	pop = (double *)malloc(DIM*sizeof(double));
	pp_min = (double *)malloc(Npp*sizeof(double));
	pp_max = (double *)malloc(Npp*sizeof(double));
	pp_step = (double *)malloc(Npp*sizeof(double));
	f = (double *)malloc(DIM*sizeof(double));
	umin = (double *)malloc(sizeof(double));
	umax = (double *)malloc(sizeof(double));
	ustep = (double *)malloc(sizeof(double));
	
	bins = (long *)malloc(N_BIN*N_BIN*sizeof(long));
	ubins = (long *)malloc(N_BIN*sizeof(long));
	
	// initial populations
	for( j = 0; j < DIM; j++ ) {
		pop[j] = 1.0;
	}
	pp_min[0] = INFTY;
	pp_max[0] = 0.0;
	pp_min[1] = INFTY;
	pp_max[1] = 0.0;

	// initial run
 	CPUbegin=clock(); // start time measurement
 	
	initial_run(NSKIP,f,pop,pp_min,pp_max,umin,umax);
	
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("The initial run: %.0e steps, CPU time = %g\n",NSKIP, cpu);

	// enlarge the bounding box
	for( j = 0; j < Npp; j++ ){
		pp_min[j] *= 0.5;
		pp_max[j] *= 1.2;
	}
	aux = 1.5*max(fabs((*umin)),(*umax));
	*umin = -aux;
	*umax = aux;
	
	// set up the domain
	for( j = 0; j < Npp; j++ ) {
		pp_step[j] = (pp_max[j] - pp_min[j])/(N_BIN - 1);
	}
	*ustep = ((*umax) - (*umin))/(N_BIN - 1);
	
	printf("The bounding box:\n");
	printf("prey_min = %.4e, prey_max = %.4e\n",pp_min[0],pp_max[0]);
	printf("predator_min = %.4e, predator_max = %.4e\n",pp_min[1],pp_max[1]);
	printf("u_min = %.4e, u_max = %.4e\n",*umin,*umax);
		
	// initialize bins
	for( j = 0; j < N_BIN; j++ ) ubins[j] = 0;
	for( j = 0; j < Nbins; j++ ) bins[j] = 0;
	
	// bin the trajectory
 	CPUbegin=clock(); // start time measurement	
 	
	binning_trajectory(NSTEPS,bins,ubins,f,pop,pp_min,pp_step,umin,ustep);
	
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("The binning run: %.0e steps, CPU time = %g\n",NSTEPS,cpu);
	
	//--------- write bins to files
	char fname[100];
	FILE *fid;
	int i;
		
	sprintf(fname,"Data/bins_alpha%.3f_sigma%.3f_gbar%.4f.txt",ALPHA,SIGMA,gbar);
	fid = fopen(fname,"w");
	for( i=0; i<N_BIN; i++ ) {
		for( j=0; j<N_BIN; j++ ) {
			ind = i + j*N_BIN;
			fprintf(fid,"%li\t",bins[ind]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
	
	sprintf(fname,"Data/ubins_alpha%.3f_sigma%.3f.txt",ALPHA,SIGMA);
	fid = fopen(fname,"w");
	for( ind = 0; ind < N_BIN; ind++ ) {
		fprintf(fid,"%li\n",ubins[ind]);
	}
	fclose(fid);
	
	sprintf(fname,"Data/params.txt");
	fid = fopen(fname,"w");
	fprintf(fid,"ALPHA = %.3f, SIGMA = %.3f, GBAR = %.4f\n",ALPHA,SIGMA,gbar);
	fprintf(fid,"N_BIN = %i\n",N_BIN);
	fprintf(fid,"prey_min = %.4e, prey_max = %.4e\n",pp_min[0],pp_max[0]);
	fprintf(fid,"predator_min = %.4e, predator_max = %.4e\n",pp_min[1],pp_max[1]);
	fprintf(fid,"umin = %.4e, umax = %.4e\n",*umin,*umax);
	fclose(fid);
	
	free(pop);
	free(f);
	free(pp_min);
	free(pp_max);
	free(pp_step);
	free(umin);
	free(umax);
	free(ustep);
	free(bins);
	free(ubins);

	return 0;
}
