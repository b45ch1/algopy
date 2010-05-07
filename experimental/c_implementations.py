import instant

c_code_adouble__imul__ = """
void mul( int tmp_lhs, double *lhs, int Ndim_lhs, int * Dims_lhs, double *lhs_tc, double rhs, int Ndim_rhs, int * Dims_rhs, double *rhs_tc){
	if(Ndim_lhs == 1){
		const int D = Dims_lhs[0];
		const int E = Dims_rhs[0];
		for(int d=D-1; d >= 0; --d){
			lhs_tc[d] *= rhs;
			const int e = (0<=d-E)*(d-E);
			for(int k = e; k < d; ++k){
				lhs_tc[d] += lhs_tc[k] * rhs_tc[d-1-e-k];
			}
			if(d<E){
				lhs_tc[d] += lhs[0] * rhs_tc[d];
			}
		}
		lhs[0]*= rhs;
	}
	else if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int E = Dims_rhs[0];
		const int Ndir = Dims_lhs[1];

		for(int d=D-1; d >= 0; --d){
			for(int n = 0; n != Ndir; ++n){
				lhs_tc[d*Ndir + n] *= rhs;
			}
			const int e = (0<=d-E)*(d-E);
			for(int k = e; k < d; ++k){
				for(int n = 0; n != Ndir; ++n){
					lhs_tc[d*Ndir+n] += lhs_tc[k*Ndir + n] * rhs_tc[(d-1-e-k)*Ndir+n];
				}
			}
			if(d<E){
				for(int n = 0; n != Ndir; ++n){
					lhs_tc[d*Ndir+n] += lhs[0] * rhs_tc[d*Ndir+n];
				}
			}
		}
		lhs[0]*= rhs;
	}	
}
"""



c_code_adouble__add__ = """
void add(int Ndim_lhs, int * Dims_lhs, double *lhs, int Ndim_rhs, int * Dims_rhs, double *rhs, int Ndim_result, int * Dims_result, double *result){
	if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int Ndir = Dims_lhs[1];
		for(int d=0; d != D; ++d){
			for(int n=0; n != Ndir; ++n){
				result[d*Ndir + n] += lhs[d*Ndir+n];
				result[d*Ndir + n] += rhs[d*Ndir+n];
			}
		}
	}
	else{
		const int D = Dims_lhs[0];
		for(int d=0; d != D; ++d){
			result[d] += lhs[d];
			result[d] += rhs[d];
		}
	}
}
"""

c_code_adouble__sub__ = """
void sub(int Ndim_lhs, int * Dims_lhs, double *lhs, int Ndim_rhs, int * Dims_rhs, double *rhs, int Ndim_result, int * Dims_result, double *result){
	if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int Ndir = Dims_lhs[1];
		for(int d=0; d != D; ++d){
			for(int n=0; n != Ndir; ++n){
				result[d*Ndir + n] -= lhs[d*Ndir+n];
				result[d*Ndir + n] -= rhs[d*Ndir+n];
			}
		}
	}
	else{
		const int D = Dims_lhs[0];
		for(int d=0; d != D; ++d){
			result[d] -= lhs[d];
			result[d] -= rhs[d];
		}
	}
}
"""

c_code_adouble__mul__ = """
void mul(int Ndim_lhs, int * Dims_lhs, double *lhs, int Ndim_rhs, int * Dims_rhs, double *rhs, int Ndim_result, int * Dims_result, double *result){
	if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int Ndir = Dims_lhs[1];
		for(int d=0; d != D; ++d){
			for(int k = 0; k != d; ++k){
				for(int n=0; n != Ndir; ++n){
					result[d*Ndir + n] += lhs[k*Ndir+n] * rhs[(d-k)*Ndir+n];
				}
			}
		}
	}
	else{
		const int D = Dims_lhs[0];
		for(int d=0; d != D; ++d){
			for(int k = 0; k <= d; ++k){
				result[d] += lhs[k] * rhs[d-k];
			}
		}
	}
}
"""

c_code_adouble__div__ = """
void div(int Ndim_lhs, int * Dims_lhs, double *lhs, int Ndim_rhs, int * Dims_rhs, double *rhs, int Ndim_result, int * Dims_result, double *result){
	if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int Ndir = Dims_lhs[1];
		for(int d=0; d != D; ++d){
			for(int n=0; n != Ndir; ++n){
				result[d*Ndir+n]+=lhs[d*Ndir+n];
			}
			for(int k=0; k!=d; ++k){
				for(int n=0; n != Ndir; ++n){
					result[d*Ndir + n] -= result[k*Ndir + n]*rhs[(d-k)*Ndir + n];
				}
			}
			for(int n=0; n != Ndir; ++n){
				result[d*Ndir + n] /= rhs[0+n];
			}

		}
	}
	else{
		const int D = Dims_lhs[0];
		for(int d=0; d != D; ++d){
			result[d]+=lhs[d];
			for(int k=0; k!=d; ++k){
				result[d] -= result[k]*rhs[d-k];
			}
			result[d] /= rhs[0];
		}
	}
}
"""

adouble__add__ = instant.inline_with_numpy(c_code_adouble__add__, arrays=[['Ndim_lhs', 'Dims_lhs', 'lhs'], ['Ndim_rhs', 'Dims_rhs', 'rhs'], ['Ndim_result', 'Dims_result', 'result']] )
adouble__sub__ = instant.inline_with_numpy(c_code_adouble__sub__, arrays=[['Ndim_lhs', 'Dims_lhs', 'lhs'], ['Ndim_rhs', 'Dims_rhs', 'rhs'], ['Ndim_result', 'Dims_result', 'result']] )
adouble__mul__ = instant.inline_with_numpy(c_code_adouble__mul__, arrays=[['Ndim_lhs', 'Dims_lhs', 'lhs'], ['Ndim_rhs', 'Dims_rhs', 'rhs'], ['Ndim_result', 'Dims_result', 'result']] )
adouble__div__ = instant.inline_with_numpy(c_code_adouble__div__, arrays=[['Ndim_lhs', 'Dims_lhs', 'lhs'], ['Ndim_rhs', 'Dims_rhs', 'rhs'], ['Ndim_result', 'Dims_result', 'result']] )
adouble__imul__ = instant.inline_with_numpy(c_code_adouble__imul__, arrays=[['tmp_lhs','lhs'],['Ndim_lhs', 'Dims_lhs', 'lhs_tc'], ['Ndim_rhs', 'Dims_rhs', 'rhs_tc']] )
 
