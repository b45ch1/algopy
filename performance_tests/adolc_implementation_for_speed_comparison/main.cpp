#include "adolc/adolc.h"
#include <vector>
#include <sys/time.h>


struct timeval tv;
int mtime(void){
	gettimeofday(&tv,NULL);  
	return (int)(tv.tv_sec*1000 + (tv.tv_usec / 1000));
}


template<class Tdouble>
Tdouble f(Tdouble *x,int N){
	Tdouble tmp = 1.;
	for(int n = 0; n != N; ++n){
		tmp *=x[n];
	}
	return tmp;
}

int main( int argc, char *argv ){
	vector<int> Ns(10);
	
	
	const int N = 100;
	const int P = N*(N+1)/2;
	const int D = 3;
	const int M = 1;
	double x[N];
	double y;
	for(int n = 0; n!=N; ++n){
		x[n] = n+2;
	}
	printf("x=\n");
	for(int n = 0; n!=N; ++n){
		printf("%f ",x[n]);
	}
	printf("\n");
// 	printf("speelpenning(x,N)= %f\n",speelpenning(x,N));

	int start_time;
	int end_time;
	start_time = mtime();
	adouble ax[N];
	adouble ay;

	trace_on(0);
		ay=1.;
		for(int n = 0; n < N; ++n ){
			ax[n]<<=x[n];
			ay *= ax[n];
		} 
		ay>>=y;
	trace_off();


// 	/* hos_forward */
// 	double **X;
// 	X = myalloc2(N,D);
// 	X[0][0]=1.;
// 	X[1][0]=1.;
// 	for(int n=0; n!=N; ++n){
// 		for(int d=0; d!=D; ++d){
// 			printf("%f",X[n][d]);
// 		}
// 		printf("\n");
// 	}
// 	double **Y;
// 	Y = myalloc(1,D);
// 	hos_forward(0,M,N,D,0,x,X,&y,Y);
// 	for(int m=0; m!=M; ++m){
// 		for(int d=0; d!=D; ++d){
// 			printf("%f ",Y[m][d]);
// 		}
// 		printf("\n");
// 	}

	

	/* hov_forward */
	double ***X;
	X = myalloc3(N,P,D);

	
// 	for(int n = 0; n!=N; ++n){
// 		for(int p = 0; p!=P; ++p){
// 			for(int d = 0; d != D; ++d){
// 				X[n][p][d] = 0.;
// 				if(d==0){
// 					X[n][p][d] = (n==p);
// 				}
// 			}
// 		}
// 	}
// 
// 	printf("X=\n");
// 	for(int n = 0; n!=N; ++n){
// 		for(int p = 0; p!=P; ++p){
// 			for(int d = 0; d != D; ++d){
// 				printf("%f ",X[n][p][d]);
// 			}
// 			printf("\n");
// 		}
// 		printf("\n");
// 	}

	double ***Y;
	Y = myalloc3(1,P,D);

// 	zos_forward(0,1,N,0,x,&y);
// 	printf("y= %f\n",y);
	hov_forward(0,1,N,D,P,x,X,&y,Y);

// 	printf("Y=\n");
// 	for(int m = 0; m != M; ++m){
// 		for(int p = 0; p!=N; ++p){
// 			for(int d = 0; d != D; ++d){
// 			printf("%f ",Y[0][p][d]);
// 			}
// 		printf("\n");
// 		}
// 		printf("\n");
// 	}
end_time = mtime();
double run_time = (end_time-start_time)/1000.;
printf("required time = %0.6f\n",run_time);

/* reverse mode speed */
// 	double** H;
// 	H = myalloc2(N,N);
// 	
// 	start_time = mtime();
// 	hessian(0,N,x,H);
// 	end_time = mtime();
// 	double run_time = (end_time-start_time)/1000.;
// 
// 	printf("required time = %0.6f\n",run_time);
	
// 	for(int n1=0; n1!=N; ++n1){
// 		for(int n2=0; n2!=N; ++n2){
// 			printf("%f ", H[n1][n2]);
// 		}
// 		cout<<std::endl;
// 	}



	return 0;
}
