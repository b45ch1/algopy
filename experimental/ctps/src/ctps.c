/*============================================================================
Description: Evaluating Cross-Derivatives for functions in math.h

Authors: Andreas Griewank
        Lutz Lehmann
        Marat Zilberman

Version of: 16.09.2009
General Comments: Some of the implemented functions aren't differentiable or
                    undefined at some points. We assume that their derivatives are
                    evaluated at differentiable points.
                    Although many functions have the same runtime complexity, their
                    actual runtime may differ, we have used clock ticks to measure
                    actual runtime. To see difference in clock ticks use
                    15 variables or more as an input to the main() function.
Problems/bugs: Composition of ctps_mul() with ctps_div() is not accurate
                for 16 varibles or more, which is relatively a bad result.
                Other compositions work well. Some are very good like ctps_exp
                and natural logarithm, ctps_square and squareroot, for 20 variables
                they show good results.
============================================================================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h> /* for tests (ticks measurement) */


/* EPSILON - precision value, used in the test program.
PI - PI constant */
#define EPSILON (0.0000000001)
#define PI 3.14159265358979323846264338327950288


/*============================================================================*/
void ctps_add(int h, double* u, double* w, double* v){
    /* Implementation of v=u+w.
    Run-time: O(2^n).
    Storage: O(1).*/

    int i;
    for(i=0;i<h;i++){
        v[i] = u[i]+w[i];
    }
}
/*============================================================================*/
void ctps_sub(int h, double* u, double* w, double* v){
    /* Implementation of v=u-w.
    Run-time: O(2^n).
    Storage: O(1).*/

    int i;
    for(i=0;i<h;i++){
        v[i] = u[i]-w[i];
    }
}
/*============================================================================*/


/*==========================================================================*/
void ctps_mul(int h, double*u, double*w, double* v)
{
    /* This version is preferable because the base condition
    saving recursive function calls.
    Runtime: O(3^n)
    Storage: O(n). */
    
    if (h==4) {
        v[3] += (u[3]*w[0]+u[2]*w[1]+u[1]*w[2]+u[0]*w[3]);
        v[2] += (u[2]*w[0]+u[0]*w[2]);
        v[1] += (u[1]*w[0]+u[0]*w[1]);
        v[0] += (u[0]*w[0]);
        return;
    }
    if (h==2) {
        v[1] += (u[1]*w[0]+u[0]*w[1]);
        v[0] += (u[0]*w[0]);
        return;
    }
    if (h==1) {
        v[0] += (u[0]*w[0]);
        return;
    }
    if (h==-4) {
        v[0] -= (u[0]*w[0]);
        v[1] -= (u[1]*w[0]+u[0]*w[1]);
        v[2] -= (u[2]*w[0]+u[0]*w[2]);
        v[3] -= (u[3]*w[0]+u[2]*w[1]+u[1]*w[2]+u[0]*w[3]);
        return;
    }
    if (h==-2) {
        v[0] -= (u[0]*w[0]);
        v[1] -= (u[1]*w[0]+u[0]*w[1]);
        return;
    }
    if (h==-1) {
        v[0] -= (u[0]*w[0]);
        return;
    }
    
    h /= 2;
    if(h > 0) {
        ctps_mul(h,u,w+abs(h),v+abs(h));
        ctps_mul(h,u+abs(h),w,v+abs(h));
        ctps_mul(h,u,w,v);
    } else {
        ctps_mul(h,u,w,v);
        ctps_mul(h,u+abs(h),w,v+abs(h));
        ctps_mul(h,u,w+abs(h),v+abs(h));
    }
}
/*==========================================================================*/
void ctps_div(int h, double* u, double* w, double* v){
    /* Implementation of v=u/w using deconvolutions. Writing result into v.
    Runtime: O(3^n)
    Storage: O(n) */
    
    int i,j;
    double w0;
    
    w0 = *w;
    for(j=0;j<h;j++) {
        v[j] = u[j] / w0;
        w[j] /= w0;
    }
    *w = 0;
    
    /* i=1: v[1]-=w[1]*v[0]; v[1]-=w[0]*v[1]; but w[0]==0
        unnormalized v[1]=u[1]/w[0]-w[1]*u[0]/(w[0]*w[0]); */
    
    for(i=1;i<h;i*=2) {
        ctps_mul(-i,w+i,v,v+i);
        ctps_mul(-i,w,v+i,v+i);
    }
    for(j=1;j<h;j++) {
        w[j] *= w0;
    }
    *w = w0;
}
/*==========================================================================*/
void ctps_iadd(int h, double* u, double* v){
    /* Implementation of v+=u
    Run-time: O(2^n).
    Storage: O(1).*/

    int i;
    for(i=0;i<h;i++){
        v[i] += u[i];
    }
}
/*==========================================================================*/
void ctps_isub(int h, double* u, double* v){
    /* Implementation of v-=u
    Run-time: O(2^n).
    Storage: O(1).*/

    int i;
    for(i=0;i<h;i++){
        v[i] -= u[i];
    }
}
/*==========================================================================*/
void ctps_imul(int h, double* u, double* v)
{
/* 2 operand version of multplication. Computes derivatives
of v*=u and writes them into v.
Storage: O(n)
Runtime: O(3^n)
*/

int k=h;
int c=(h>0)?1:-1;

while (abs(k)>2) {
    k/=2;
    ctps_imul (k,u,v+abs(k));
    ctps_mul (k,u+abs(k),v,v+abs(k));
}
if(abs(h)>1) {
    v[1]*=(c*u[0]);
    v[1]+=(c*u[1]*v[0]);
}
v[0]*=(c*u[0]);
}
/*==========================================================================*/
void ctps_idiv (int h, double* u, double* v){
    /* 2 operand version of division. Computes derivatives
    of v /= u
    Runtime: O(3^n)
    Storage: O(n).  */
    
    int k;
    double u0=u[0];
    
    for(k=0; k<h; k++){
        u[k] /= u0;
        v[k] /= u0;
    }
    if (h==1) {
        u[0]=u0;
        return;
    }
    v[1]-=v[0]*u[1];
    for (k=2; k<h; k*=2) {
        ctps_mul (-k,u+k,v,v+k);
        ctps_idiv (k,u,v+k);
    }
    for(k=0; k<h; k++){
        u[k] *= u0;
    }
}
/*==========================================================================*/
void ctps_inv (int h, double* u){
    /* 1 operand version of reciprocal. Computes derivatives
    of 1/u and writes them into u.
    Runtime: O(3^n).
    Storage: O(n) */
    
    int i,k=h;
    
    u[0]=1/u[0];
    if (h==1) return;
    u[1]=-u[0]*u[0]*u[1];
    
    for(k=2; k<h; k*=2) {
        ctps_imul (k,u,u+k);
        ctps_imul (k,u,u+k);
        for (i=0; i<k; i++) {
            u[k+i]=-u[k+i];
        }
    }
}
/*==========================================================================*/
void ctps_exp(int h, double* u, double* v)
{
/* Implementation of v=exp(u) based on its ODE.
Runtime: O(3^n)
Storage: O(n) */

int i;
for(i=0;i<h;i++) v[i] = 0.0;
*v = exp(*u);
for(i=1;i<h;i*=2)
    ctps_mul(i, v, u+i, v+i);
}
/*==========================================================================*/
void ctps_log(int h, double* u, double* v)
{
/* Implementation of v=ln(u) based on its ODE.
Runtime: O(3^n)
Storage: O(n) */

int i;
double u0;

*v = log(*u);
for(i=1;i<h;i++) {
    v[i] = u[i] /= *u;
}
u0 = *u;
*u = 0;
for(i=1;i<h;i*=2) {
    ctps_mul(-i,u,v+i,v+i);
}
*u = u0;
for(i=1;i<h;i++) {
    u[i] *= *u;
}
}
/*==========================================================================*/
void ctps_log10(int h, double* u, double* v)
{
/* Implementation of v=log10(u) based on ctps_log().
Runtime: O(3^n)
Storage: O(n) */

int i;
double c=1/log(10);

ctps_log(h,u,v);
for(i=0;i<h;i++) {
    v[i] *= c;
}
}
/*==========================================================================*/
void trigon(int h, double* u, double* sine, double* cose)
{
/* Implementation of sine=sin(u) and cose=cos(u) based on
their mutual ODE.
Runtime: O(3^n)
Storage: O(n) */

int i;

*sine = sin(*u);
*cose = cos(*u);
for(i=1;i<h;i++) {
    sine[i] = 0.0;
    cose[i] = 0.0;
}
for(i=1;i<h;i*=2) {
    ctps_mul(i,cose,u+i,sine+i);
    ctps_mul(-i,sine,u+i,cose+i);
}
}
/*==========================================================================*/
void ctps_tangent (int h, double* u, double* v)
{
/* Implementation of v=tan(u) based on its ODE,
and then applying recursive rule, using the
structure of partials array
Runtime: O(3^n)
Storage: O(n) */

int k,i;
double* w=v+h/2;

v[0]=tan(u[0]);
if (h==1) return;
w[0]=1+v[0]*v[0];
v[1]=w[0]*u[1];
if (h==2) return;
w[1]=2*v[0]*v[1];

for (k=2; k<h/2; k*=2) {
    ctps_mul(k,w,u+k,v+k);
    ctps_mul(k,v,v+k,w+k);
    for (i=0; i<k; i++) {
        w[k+i]*=2;
    }
}
ctps_imul (h/2,u+h/2,v+h/2);
}
/*==========================================================================*/
void tangent_ver2 (int h, double* u, double* v)
{
/* Implementation of v=tan(u) based on tangents definition,
Storage: O(2^n)
Runtime: O(3^n) */

double* sine = (double*) calloc(h,sizeof(double));
double* cose = (double*) calloc(h,sizeof(double));

trigon(h,u,sine,cose);
ctps_div (h,sine,cose,v);

free(sine);
free(cose);
}
/*==========================================================================*/
void ctps_pow(int h, double r, double* u, double* v)
{
/* Implementation of v=u^r. r-constant passed to
the function. based on its ODE.
Runtime: O(3^n)
Storage: O(n) */

int i,j;
double u0;

*v = pow(*u,r);
for(j=1;j<h;j++) {
    u[j] /= *u;
    v[j] = 0;
}
u0 = *u;
*u = 0;
for(i=1;i<h;i*=2) {
        ctps_mul(i,v,u+i,v+i);
        for(j=i;j<2*i;j++) {
        v[j] *= r;
        }
        ctps_mul(-i,u,v+i,v+i);
}
*u = u0;
for(i=1;i<h;i++) {
    u[i] *= u0;
}
}
/*==========================================================================*/
void ctps_square(int h, double* u, double* v)
{
/* Implementation of v=u^2 based on its ODE.
Runtime: O(3^n)
Storage: O(n) */

int i,j;
*v = *u * *u;
for(j=1;j<h;j++) {
    v[j] = 0;
}
for(i=1;i<h;i*=2) {
    ctps_mul(i,u,u+i,v+i);
}
for(j=1;j<h;j++) {
    v[j] *= 2;
}
}
/*==========================================================================*/
void ctps_sqrt(int h, double* u, double* v)
{
/* Implementation of v=sqrt(u) based on its ODE.
Runtime: O(3^n)
Storage: O(n) */

int i;

*v =0;
for(i=1;i<h;i++) {
    v[i] = 0.5*u[i]/ *u;
}
for(i=1;i<h;i*=2) {
    ctps_mul(-i,v,v+i,v+i);
}
*v = sqrt(*u);
for(i=1;i<h;i++) {
    v[i] *= *v;
}
}
/*==========================================================================*/

void trigon_hyp(int h, double* u, double* sin_hyp, double* cos_hyp)
{
/* Implementation of sin_hyp=sinh(u) and cos_hyp=cosh(u) based on
their mutual ODE.
Runtime: O(3^n)
Storage: O(n) */

int i;

*sin_hyp = sinh(*u);
*cos_hyp = cosh(*u);
for(i=1;i<h;i++) {
    sin_hyp[i] = 0.0;
    cos_hyp[i] = 0.0;
}
for(i=1;i<h;i*=2) {
    ctps_mul(i,cos_hyp,u+i,sin_hyp+i);
    ctps_mul(i,sin_hyp,u+i,cos_hyp+i);
}
}
/*==========================================================================*/

void tangent_hyp (int h, double* u, double* v)
{
/* Implementation of v=tanh(u) based on its ODE,
and then applying recursive rule, using the
structure of partials array.
Runtime: O(3^n)
Storage: O(n) */

int k,i;
double* w=v+h/2;

v[0]=tanh(u[0]);
if (h==1) return;
w[0]=1-v[0]*v[0];
v[1]=w[0]*u[1];
if (h==2) return;
w[1]=(-2)*v[0]*v[1];

for (k=2; k<h/2; k*=2) {
    ctps_mul(k,w,u+k,v+k);
    ctps_mul(k,v,v+k,w+k);
    for (i=0; i<k; i++) {
        w[k+i]*=(-2);
    }
}
ctps_imul (h/2,u+h/2,v+h/2);
}
/*==========================================================================*/

void tangent_hyp_ver2 (int h, double* u, double* v)
{
/* Implementation of v=tan_hyp(u) based on its definition,
Storage: O(2^n)
Runtime: O(3^n) */

double* sin_hyp = (double*) calloc(h,sizeof(double));
double* cos_hyp = (double*) calloc(h,sizeof(double));

trigon_hyp(h,u,sin_hyp,cos_hyp);
ctps_div (h,sin_hyp,cos_hyp,v);

free(sin_hyp);
free(cos_hyp);
}
/*==========================================================================*/

void arccos(int h, double* u, double* v)
{
/* Implementation of v=arccos(u) based on its first derivative,
then applying recursive rule, using the structure of
partials array.
Runtime: O(3^n)
Storage: O(n) */

int i,k;
double* w=v+h/2;

v[0]=acos(u[0]);
if (h==1) return;
w[0]=sqrt(1-u[0]*u[0]);
v[1]=-u[1]/w[0];
if (h==2) return;
w[1]=v[1]*u[0];

for (k=2; k<h/2; k*=2) {
    ctps_div (k,u+k,w,v+k);
    for (i=0; i<k; i++) {
        v[k+i]*=-1;
    }
    ctps_mul(k,u,v+k,w+k);
}
ctps_inv (h/2,w);
for (i=0; i<h/2; i++) {
    w[i]*=-1;
}
ctps_imul(h/2,u+h/2,w);
}
/*==========================================================================*/

void arcsin (int h, double* u, double* v)
{
/* Implementation of v=arcsin(u) based on arccos.
Runtime: O(3^n)
Storage: O(n) */

int i;

arccos(h,u,v);
v[0]=asin(u[0]);
for (i=1; i<h; i++) {
    v[i]=-v[i];
}
}
/*==========================================================================*/

void arctan (int h, double* u, double* v)
{
/* Implementation of v=arctan(u) based on its first derivative,
then applying recursive rule, using the structure of
partials array.
Runtime: O(3^n)
Storage: O(n) */

int i,k;
double *w=v+h/2;

v[0]=atan(u[0]);
if (h==1) return;
w[0]=1+u[0]*u[0];
v[1]=u[1]/w[0];
if (h==2) return;
w[1]=2*u[0]*u[1];

for (k=2; k<h/2; k*=2) {
    ctps_div (k,u+k,w,v+k);
    ctps_mul(k,u,u+k,w+k);
    for (i=0; i<k; i++) {
        w[k+i]*=2;
    }
}
ctps_inv (h/2,w);
ctps_imul(h/2,u+h/2,w);
}
/*==========================================================================*/
void arctan2 (int h, double* y, double* x, double* v)
{
/* Implementation of v=arctan2(y,x), based on arctan(y/x), and partition into
cases.
Runtime: O(3^n)
Storage: O(2^n) */

int i;
double *temp= (double*) calloc(h,sizeof(double));

if (fabs(y[0]/x[0])<1)
{
    ctps_div (h,y,x,temp);
    arctan (h,temp,v);
    if (x[0]<0)
    {
        if (y[0]>0)
            v[0]+=PI;
        if (y[0]<0)
            v[0]-=PI;
        if (y[0]==0)
        v[0]=PI;
    }
}
else
{
    ctps_div (h,x,y,temp);
    arctan (h,temp,v);
    if (v[0]>0) {
        v[0]=PI/2-v[0];
    }
    else {
        v[0]=-PI/2-v[0];
    }
    for (i=1; i<h; i++) {
        v[i]=-v[i];
    }
}
free(temp);
}

/*==========================================================================*/
void ctps_fabs (int h, double* u, double* v)
{
/* Implementation of v=fabs(u). depends on u[0] value.
Runtime: O(2^n)
Storage: O(1) */

int i,c;

v[0]=fabs(u[0]);
c=(u[0]>=0)?1:-1;
for (i=1; i<h; i++)
    v[i]=c*u[i];
}
/*==========================================================================*/

void floorceil (int h, double* u, double* vfloor, double* vceil)
{
/* Implementation of vfloor=floor(u) and vceil=ceil(u).
Runtime: O(2^n)
Storage: O(1) */

int i;

vfloor[0]=floor(u[0]);
vceil[0]=ceil(u[0]);

for (i=1; i<h; i++) {
    vfloor[i]=0;
    vceil[i]=0;
}
}
/*==========================================================================*/

void moduluf (int h, double* u, double* v)
{
/* Implementation of v=modf(u) (fractional part of u with the same sign).
Runtime: O(2^n)
Storage: O(1) */

int i;
double intpart; /*technical variable*/

v[0]=modf(u[0],&intpart);
for (i=1; i<h; i++)
    v[i]=u[i];
}
/*==========================================================================*/

void raisepow (int h, double* u, double* w, double* v)
{
/* Implementation of v=u^w, based on taking natural logarithm
of both sides of the equation.
Storage: O(2^n)
Runtime: O(3^n) */

int i;
double *arr1, *arr2;

arr1 = (double*) calloc(h,sizeof(double));
arr2 = (double*) calloc(h,sizeof(double));

for (i=0; i<h; i++) {
    arr2[i]=0;
}
ctps_log (h,u,arr1);
ctps_mul(h,w,arr1,arr2);
ctps_exp (h,arr2,v);

free (arr1);
free (arr2);
}
/*==========================================================================*/

void frexponent (int h, double* u, double* v)
{
/* Implementation of v=frexp(u) (returns binary significand of u). Depends
in which interval of squares U[0] is.
Storage: O(1)
Runtime: O(2^n) */

int exp_val,i;
double c;

v[0]=frexp(u[0],&exp_val);
c=pow(2,-exp_val);
for (i=1; i<h; i++) {
    v[i]=c*u[i];
}
}
/*==========================================================================*/

void ldexponent (int h, double* u, double exp, double* v)
{
/* Implementation of v=ldexp(u,exp) (v=u*(2^exp)).
exp is a constant passed to a function.
Storage: O(1)
Runtime: O(2^n) */

double k=pow(2,exp);
int i;

for (i=0; i<h; i++) {
    v[i]=k*u[i];
}
}
/*==========================================================================*/

void fmodulo (int h, double* u, double* w, double* v)
{
/* Implementation of v=fmod(u,w) - remainder of u/w.
Runtime: O(2^n)
Storage: O(1) */

int i;
double k;

v[0]=fmod(u[0],w[0]);
k=(u[0]-v[0])/w[0];
for (i=1; i<h; i++)
    v[i]=u[i]-k*w[i];
}

/*==========================================================================
Here are functions dealing with crossmultiplication, there are several versions
of these functions, see comment on each function. All functions can propogate
partials of v=u*w and are at least of O(3^n) run-time complexity (n - number of
independent variables). Tests have shown (based on clock ticks),that mul()
and ctps_imul() are the fastest.
IMPORTANT: crossmult, mul, ctps_imul can handle deconvolution.
            mult, smartmult - cannot handle deconvolution.
============================================================================*/
void crossmult(int h, double* u, double* w, double* v){
    /* crossmultiplies the first h ( power of 2 ) elments of u with
    those of w and increments the entries of v by it.
    printf(" h=%d,  i= %d , j= %d count = %d \n",h, i,j,count );
    Runtime: O(3^n)
    Storage: O(n).*/

    int i;
    int c=(h>0?1:-1);
    *v += c* *u* *w;
    if(abs(h)>1)
        v[1]+=c* (u[0]*w[1]+u[1]*w[0]);
    /*loop*/
    for (i=2; i<abs(h); i*=2) {
        crossmult((h>0?i:-i), u, w+i, v+i);
        crossmult((h>0?i:-i), u+i, w, v+i);
    }
}

/*==========================================================================*/
void mult (int h, double* u, double* w, double* v){
    /* This version is the non-recursive implementation of multiplicaton.
    Storage: O(1), However, run-time is longer (additional branch
    checks). Runtime: At least O(3^n). Note the analogy between the indices
    and bitwise operations to sets and set operations.
    IMPORTANT - cannot calculate deconvolution. */
    
    int i,j,t,delta; /* i,j are indices over the u,w arrays. delta is
                        the step for j'th index (j is not always incremented
                        by 1), t is the intersection index of i and j. */
    v[0]+=w[0]*u[0];
    
    for (i=0; i<h; i++) {
    
        delta=1;
        while ((delta&i)!=0) {delta*=2;}
    
        j=0;
        while (j<i) {
            t=i&j;
            if (t==0) {
                v[i|j]+=(u[i]*w[j] + u[j]*w[i]);
                j+=delta;
            }
            else {
                j+=t;
            }
        }
    }
}

/*==========================================================================*/
void smartmult (int h, double* u, double*w, double*v)
{
    /* The base condition saving recursive function calls - it uses loop.
    IMPORTANT - cannot calculate deconvolution.
    Storage: O(n)
    Runtime: O(3^n) */
    
    if( h<= 16)
        mult (h,u,w,v);
    else{
        h /= 2;
        smartmult(h,u,w,v);
        smartmult(h,u+abs(h),w,v+abs(h));
        smartmult(h,u,w+abs(h),v+abs(h));
    }
}
/*==========================================================================*/

