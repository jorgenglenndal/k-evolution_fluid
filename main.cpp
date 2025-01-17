//////////////////////////
// Copyright (c) 2015-2019 Julian Adamek/ 2019 -2023 Farbod Hassani
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//////////////////////////

//////////////////////////
// main.cpp
//////////////////////////
//
// main control sequence of k-evolution N-body code based on gevolution.
// Author (k-evolution): Farbod Hassani (Université de Genève & Universitetet i Oslo)
// Author (gevolution): Julian Adamek  (Université de Genève & Observatoire de Paris & Queen Mary University of London)
//
// Last modified: Feb 2 2023  by Farbod Hassani
//
//////////////////////////
#include <stdlib.h>
#include <set>
#include <vector>
#ifdef NONLINEAR_TEST
#include <iomanip>
#include<fstream>
#include<sstream>
#include <limits> // Include the <limits> header
#include <cmath>
#endif
#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
#include "class.h"
#undef MAX			// due to macro collision this has to be done BEFORE including LATfield2 headers!
#undef MIN
#endif
#include "LATfield2.hpp"
#include "metadata.hpp"
#ifdef HAVE_CLASS
#include "class_tools.hpp"
#endif
#ifdef HAVE_HICLASS
#include "hiclass_tools.hpp"
#endif
#include "tools.hpp"
#include "background.hpp"
#include "Particles_gevolution.hpp"
#include "gevolution.hpp"
#include "ic_basic.hpp"
#include "ic_read.hpp"
#ifdef ICGEN_PREVOLUTION
#include "ic_prevolution.hpp"
#endif
#ifdef ICGEN_FALCONIC
#include "fcn/togevolution.hpp"
#endif
#include "radiation.hpp"
#include "parser.hpp"
#include "output.hpp"
#include "hibernation.hpp"
#ifdef VELOCITY
#include "velocity.hpp"
#endif

using namespace std;
using namespace LATfield2;

int main(int argc, char **argv)
{

#ifdef BENCHMARK
	//benchmarking variables
	double ref_time, ref2_time, cycle_start_time;
	double initialization_time;
	double run_time;
  double kessence_update_time=0; // How much time is put on updating the kessence field
	double cycle_time=0;
	double projection_time = 0;
	double snapshot_output_time = 0;
	double spectra_output_time = 0;
	double lightcone_output_time = 0;
	double gravity_solver_time = 0;
	double fft_time = 0;
	int fft_count = 0;
	double update_q_time = 0;
	int update_q_count = 0;
	double moveParts_time = 0;
	int  moveParts_count =0;
	//kessence
	double a_kess;
  double Hc;
#endif  //BENCHMARK

	int n = 0, m = 0;
	int io_size = 0;
	int io_group_size = 0;
	int i, j, cycle = 0, snapcount = 0, pkcount = 0, restartcount = 0, usedparams, numparam = 0, numsteps, numspecies, done_hij;
	int numsteps_ncdm[MAX_PCL_SPECIES-2];
	long numpts3d;
	int box[3];
	double dtau, dtau_old, dx, tau, a, fourpiG, tmp, start_time;
	double maxvel[MAX_PCL_SPECIES];
	FILE * outfile;
	char filename[2*PARAM_MAX_LENGTH+24];
	string h5filename;
	char * settingsfile = NULL;
	char * precisionfile = NULL;
	parameter * params = NULL;
	metadata sim;
	cosmology cosmo;
	icsettings ic;
	double T00hom;
  #ifdef HAVE_HICLASS_BG
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  gsl_spline * H_spline = NULL;
  gsl_spline * cs2_spline = NULL;
  gsl_spline * alpha_K_spline = NULL;
  gsl_spline * cs2_prime_spline = NULL;
  gsl_spline * rho_smg_spline = NULL;
  gsl_spline * rho_cdm_spline = NULL;
  gsl_spline * rho_b_spline = NULL;
  gsl_spline * rho_g_spline = NULL;
  gsl_spline * p_smg_spline = NULL;
  gsl_spline * rho_smg_prime_spline = NULL;
  gsl_spline * p_smg_prime_spline = NULL;
  gsl_spline * rho_crit_spline = NULL;
  // testing
  gsl_spline * rho_ur_spline = NULL;
  gsl_spline * time_spline = NULL;
  gsl_spline * conformal_time_spline = NULL;
  //gsl_spline * p_tot_prime_spline = NULL;

  #endif

#ifndef H5_DEBUG
	H5Eset_auto2 (H5E_DEFAULT, NULL, NULL);
#endif

	for (i=1 ; i < argc ; i++ ){
		if ( argv[i][0] != '-' )
			continue;
		switch(argv[i][1]) {
			case 's':
				settingsfile = argv[++i]; //settings file name
				break;
			case 'n':
				n = atoi(argv[++i]); //size of the dim 1 of the processor grid
				break;
			case 'm':
				m =  atoi(argv[++i]); //size of the dim 2 of the processor grid
				break;
			case 'p':
  #if !defined(HAVE_CLASS) && !defined(HAVE_HICLASS)
				cout << "HAVE_CLASS needs to be set at compilation to use CLASS precision files" << endl;
				exit(-100);
#endif
				precisionfile = argv[++i];
				break;
			case 'i':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#endif
				io_size =  atoi(argv[++i]);
				break;
			case 'g':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#endif
				io_group_size = atoi(argv[++i]);
		}
	}

#ifndef EXTERNAL_IO
	parallel.initialize(n,m);
#else
	if (!io_size || !io_group_size)
	{
		cout << "invalid number of I/O tasks and group sizes for I/O server (-DEXTERNAL_IO)" << endl;
		exit(-1000);
	}
	parallel.initialize(n,m,io_size,io_group_size);
	if(parallel.isIO()) ioserver.start();
	else
	{
#endif


COUT << COLORTEXT_BLUE << endl;
COUT << "                                                      "<<endl;
COUT <<"KKKKKKKKK    KKKKKKK                 EEEEEEEEEEEEEEEEEEEEEE"<<endl;
COUT <<"K:::::::K    K:::::K                 E::::::::::::::::::::E"<<endl;
COUT <<"K:::::::K    K:::::K                 E::::::::::::::::::::E"<<endl;
COUT <<"K:::::::K   K::::::K                 EE::::::EEEEEEEEE::::E"<<endl;
COUT <<"KK::::::K  K:::::KKK                   E:::::E       EEEEEE"<<endl;
COUT <<"K:::::K K:::::K                      E:::::E               "<<endl;
COUT <<"K::::::K:::::K                       E::::::EEEEEEEEEE     "<<endl;
COUT <<"K:::::::::::K      ---------------   E:::::::::::::::E     volution "<<endl;
COUT <<"K:::::::::::K      -:::::::::::::-   E:::::::::::::::E     "<<endl;
COUT <<"K::::::K:::::K     ---------------   E::::::EEEEEEEEEE     "<<endl;
COUT <<"K:::::K K:::::K                      E:::::E               "<<endl;
COUT <<"KK::::::K  K:::::KKK                   E:::::E       EEEEEE"<<endl;
COUT <<"K:::::::K   K::::::K                 EE::::::EEEEEEEE:::::E"<<endl;
COUT <<"K:::::::K    K:::::K                 E::::::::::::::::::::E"<<endl;
COUT <<"K:::::::K    K:::::K                 E::::::::::::::::::::E"<<endl;
COUT <<"KKKKKKKKK    KKKKKKK                 EEEEEEEEEEEEEEEEEEEEEE"<<endl;
COUT <<COLORTEXT_RESET << endl;

// ██ ▄█▀▓█████ ██▒   █▓ ▒█████   ██▓     █    ██ ▄▄▄█████▓ ██▓ ▒█████   ███▄    █
// ██▄█▒ ▓█   ▀▓██░   █▒▒██▒  ██▒▓██▒     ██  ▓██▒▓  ██▒ ▓▒▓██▒▒██▒  ██▒ ██ ▀█   █
// ▓███▄░ ▒███   ▓██  █▒░▒██░  ██▒▒██░    ▓██  ▒██░▒ ▓██░ ▒░▒██▒▒██░  ██▒▓██  ▀█ ██▒
// ▓██ █▄ ▒▓█  ▄  ▒██ █░░▒██   ██░▒██░    ▓▓█  ░██░░ ▓██▓ ░ ░██░▒██   ██░▓██▒  ▐▌██▒
// ▒██▒ █▄░▒████▒  ▒▀█░  ░ ████▓▒░░██████▒▒▒█████▓   ▒██▒ ░ ░██░░ ████▓▒░▒██░   ▓██░
// ▒ ▒▒ ▓▒░░ ▒░ ░  ░ ▐░  ░ ▒░▒░▒░ ░ ▒░▓  ░░▒▓▒ ▒ ▒   ▒ ░░   ░▓  ░ ▒░▒░▒░ ░ ▒░   ▒ ▒
// ░ ░▒ ▒░ ░ ░  ░  ░ ░░    ░ ▒ ▒░ ░ ░ ▒  ░░░▒░ ░ ░     ░     ▒ ░  ░ ▒ ▒░ ░ ░░   ░ ▒░
// ░ ░░ ░    ░       ░░  ░ ░ ░ ▒    ░ ░    ░░░ ░ ░   ░       ▒ ░░ ░ ░ ▒     ░   ░ ░
// ░  ░      ░  ░     ░      ░ ░      ░  ░   ░               ░      ░ ░           ░
//                  ░

COUT << "running on " << n*m << " cores." << endl;

	if (settingsfile == NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no settings file specified!" << endl;
		parallel.abortForce();
	}

	COUT << " initializing..." << endl;

	start_time = MPI_Wtime();

	numparam = loadParameterFile(settingsfile, params);

	usedparams = parseMetadata(params, numparam, sim, cosmo, ic);

	COUT << " parsing of settings file completed. " << numparam << " parameters found, " << usedparams << " were used." << endl;

	sprintf(filename, "%s%s_settings_used.ini", sim.output_path, sim.basename_generic);
	saveParameterFile(filename, params, numparam);
  sprintf(filename, "%s%s_background.dat", sim.output_path, sim.basename_generic);
  outfile = fopen(filename, "w");

	free(params);

  #if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
  background class_background;
  thermo class_thermo;
  perturbs class_perturbs;

  	if (precisionfile != NULL)
    {
	  	numparam = loadParameterFile(precisionfile, params);
      if(parallel.isRoot())  COUT << " Precision parameters are found and being read!" <<" The number of precision parameters are:"<<numparam << endl;
    }
	else numparam = 0;
  #ifdef HAVE_HICLASS_BG
    //TODO_EB:add BG functions here
    initializeCLASSstructures(sim, ic, cosmo, class_background, class_thermo, class_perturbs, params, numparam);
    loadBGFunctions(class_background, H_spline, "H [1/Mpc]", sim.z_in);
    loadBGFunctions(class_background, cs2_spline, "c_s^2", sim.z_in);
    loadBGFunctions(class_background, cs2_prime_spline, "c_s^2_prime", sim.z_in);
    loadBGFunctions(class_background, rho_smg_spline, "(.)rho_smg", sim.z_in);
    loadBGFunctions(class_background, rho_cdm_spline, "(.)rho_cdm", sim.z_in);
    loadBGFunctions(class_background, rho_b_spline, "(.)rho_b", sim.z_in);
    loadBGFunctions(class_background, rho_g_spline, "(.)rho_g", sim.z_in);
    loadBGFunctions(class_background, p_smg_spline, "(.)p_smg", sim.z_in);
    loadBGFunctions(class_background, rho_smg_prime_spline, "(.)rho_smg_prime", sim.z_in);
    loadBGFunctions(class_background, p_smg_prime_spline, "(.)p_smg_prime", sim.z_in);
    loadBGFunctions(class_background, alpha_K_spline, "kineticity_smg", sim.z_in);
    loadBGFunctions(class_background, rho_crit_spline, "(.)rho_crit", sim.z_in);
	//testing
	loadBGFunctions(class_background, rho_ur_spline, "(.)rho_ur", sim.z_in);
	loadBGFunctions(class_background, time_spline, "proper time [Gyr]", sim.z_in);
	loadBGFunctions(class_background, conformal_time_spline, "conf. time [Mpc]", sim.z_in);
	//loadBGFunctions(class_background, p_tot_prime_spline, "(.)p_tot_prime", sim.z_in);
  #endif
#endif

	h5filename.reserve(2*PARAM_MAX_LENGTH);
	h5filename.assign(sim.output_path);

	box[0] = sim.numpts;
	box[1] = sim.numpts;
	box[2] = sim.numpts;

	Lattice lat(3,box,1);
	Lattice latFT;
	latFT.initializeRealFFT(lat,0);

	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_cdm;
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_b;
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_ncdm[MAX_PCL_SPECIES-2];
	Field<Real> * update_cdm_fields[3];
	Field<Real> * update_b_fields[3];
	Field<Real> * update_ncdm_fields[3];
	double f_params[5];
	set<long> IDbacklog[MAX_PCL_SPECIES];

	Field<Real> phi;
	Field<Real> source;
	Field<Real> chi;
	Field<Real> chi_old;
	Field<Real> Sij;
	Field<Real> Bi;
	Field<Cplx> scalarFT;
	Field<Cplx> SijFT;
	Field<Cplx> BiFT;
	source.initialize(lat,1);
	phi.initialize(lat,1);

	//kessence
	Field<Real> phi_old;
  //phi at two step before to compute phi'(n+1/2)
	Field<Real> phi_prime;
  #ifdef NONLINEAR_TEST
	//Field<Real> psi_prime;

  Field<Real> short_wave;
  Field<Real> relativistic_term;
  Field<Real> stress_tensor;
  Field<Real> pi_k_old;
  Field<Real> zeta_half_old;
  Field<Cplx> scalarFT_pi_old;
  Field<Cplx> scalarFT_zeta_half_old;
  
  //new

  //Field<Real> T00_Kess_test;
  //Field<Cplx> T00_KessFT_test;
  //T00_Kess_test.initialize(lat,1);
  //T00_KessFT_test.initialize(latFT,1);
  //PlanFFT<Cplx> plan_T00_Kess_test(&T00_Kess_test, &T00_KessFT_test);
  

  #ifdef FLUID_VARIABLES
  Field<Real> delta_rho_fluid;
  Field<Cplx> delta_rho_fluidFT;
  delta_rho_fluid.initialize(lat,1);
  delta_rho_fluidFT.initialize(latFT,1);
  PlanFFT<Cplx> plan_delta_rho_fluid(&delta_rho_fluid, &delta_rho_fluidFT);

  Field<Real> delta_p_fluid;
  Field<Cplx> delta_p_fluidFT;
  delta_p_fluid.initialize(lat,1);
  delta_p_fluidFT.initialize(latFT,1);
  PlanFFT<Cplx> plan_delta_p_fluid(&delta_p_fluid, &delta_p_fluidFT);

  Field<Real> div_v_upper_fluid;
  Field<Cplx> div_v_upper_fluidFT;
  div_v_upper_fluid.initialize(lat,1);
  div_v_upper_fluidFT.initialize(latFT,1);
  PlanFFT<Cplx> plan_div_v_upper_fluid(&div_v_upper_fluid, &div_v_upper_fluidFT);

  Field<Real> v_upper_i_fluid;
  Field<Cplx> v_upper_i_fluidFT;
  v_upper_i_fluid.initialize(lat,3);
  v_upper_i_fluidFT.initialize(latFT,3);
  PlanFFT<Cplx> plan_v_upper_i_fluid(&v_upper_i_fluid, &v_upper_i_fluidFT);

  Field<Real> Sigma_upper_ij_fluid;
  Field<Cplx> Sigma_upper_ij_fluidFT;
  Sigma_upper_ij_fluid.initialize(lat,3,3,symmetric);
  Sigma_upper_ij_fluidFT.initialize(latFT,3,3,symmetric);
  PlanFFT<Cplx> plan_Sigma_upper_ij_fluid(&Sigma_upper_ij_fluid, &Sigma_upper_ij_fluidFT);
//  Field<Real> v_x_fluid;
//  Field<Cplx> v_x_fluidFT;
//  v_x_fluid.initialize(lat,1);
//  v_x_fluidFT.initialize(latFT,1);
//  PlanFFT<Cplx> plan_v_x_fluid(&v_x_fluid, &v_x_fluidFT);
//
//  Field<Real> v_y_fluid;
//  Field<Cplx> v_y_fluidFT;
//  v_y_fluid.initialize(lat,1);
//  v_y_fluidFT.initialize(latFT,1);
//  PlanFFT<Cplx> plan_v_y_fluid(&v_y_fluid, &v_y_fluidFT);
//
//  Field<Real> v_z_fluid;
//  Field<Cplx> v_z_fluidFT;
//  v_z_fluid.initialize(lat,1);
//  v_z_fluidFT.initialize(latFT,1);
//  PlanFFT<Cplx> plan_v_z_fluid(&v_z_fluid, &v_z_fluidFT);
  #endif

  
  //Field<Real> Sij;
  //Field<Cplx> SijFT;
  //Field<Real> Bi;
  //Field<Cplx> BiFT;
  //Sij.initialize(lat,3,3,symmetric);
  //SijFT.initialize(latFT,3,3,symmetric);
  //PlanFFT<Cplx> plan_Sij(&Sij, &SijFT);
  //Bi.initialize(lat,3);
  //BiFT.initialize(latFT,3);
  //PlanFFT<Cplx> plan_Bi(&Bi, &BiFT);

  #endif
	Field<Real> pi_k;
  Field<Real> zeta_half;
	Field<Real> T00_Kess;
	Field<Real> T0i_Kess;
	Field<Real> Tij_Kess;
	Field<Cplx> scalarFT_phi_old;
	Field<Cplx> phi_prime_scalarFT;
  #ifdef NONLINEAR_TEST
  	Field<Real> zeta_integer;
  //Field<Cplx> psi_prime_scalarFT;
  Field<Cplx> short_wave_scalarFT;
  Field<Cplx> relativistic_term_scalarFT;
  Field<Cplx> stress_tensor_scalarFT;
  
	Field<Cplx> scalarFT_chi_old;
	Field<Cplx> scalarFT_pi;
	Field<Cplx> scalarFT_zeta_integer;
	zeta_integer.initialize(lat,1);
	scalarFT_zeta_integer.initialize(latFT,1);
	PlanFFT<Cplx> plan_zeta_integer(&zeta_integer, &scalarFT_zeta_integer);
	#endif
  Field<Cplx> scalarFT_zeta_half;
	Field<Cplx> T00_KessFT;
	Field<Cplx> T0i_KessFT;
	Field<Cplx> Tij_KessFT;
	chi.initialize(lat,1);
	scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> plan_source(&source, &scalarFT);
	PlanFFT<Cplx> plan_phi(&phi, &scalarFT);
	PlanFFT<Cplx> plan_chi(&chi, &scalarFT);
	Sij.initialize(lat,3,3,symmetric);
	SijFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_Sij(&Sij, &SijFT);
	Bi.initialize(lat,3);
	BiFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_Bi(&Bi, &BiFT);
#ifdef CHECK_B
	Field<Real> Bi_check;
	Field<Cplx> BiFT_check;
	Bi_check.initialize(lat,3);
	BiFT_check.initialize(latFT,3);
	PlanFFT<Cplx> plan_Bi_check(&Bi_check, &BiFT_check);
#endif
	//Kessence end
  #ifdef VELOCITY
  	Field<Real> vi;
  	Field<Cplx> viFT;
  	vi.initialize(lat,3);
  	viFT.initialize(latFT,3);
  	PlanFFT<Cplx> plan_vi(&vi, &viFT);
  	double a_old;
  #endif
  #ifdef NONLINEAR_TEST
  if(parallel.isRoot()) cout << "\033[1;32m The blowup tests are requested\033[0m\n";
    pi_k_old.initialize(lat,1);
  	scalarFT_pi_old.initialize(latFT,1);
  	PlanFFT<Cplx> plan_pi_k_old(&pi_k_old, &scalarFT_pi_old);
    /// zeta
    zeta_half_old.initialize(lat,1);
    scalarFT_zeta_half_old.initialize(latFT,1);
    PlanFFT<Cplx> plan_zeta_half_old(&zeta_half_old, &scalarFT_zeta_half_old);
  #endif
	//Kessence part initializing
	//Phi_old
	phi_old.initialize(lat,1);
	scalarFT_phi_old.initialize(latFT,1);
	PlanFFT<Cplx> plan_phi_old(&phi_old, &scalarFT_phi_old);
	//Phi'
	phi_prime.initialize(lat,1);
	phi_prime_scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> phi_prime_plan(&phi_prime, &phi_prime_scalarFT);
  //Relativistic corrections
  #ifdef NONLINEAR_TEST
  //psi_prime.initialize(lat,1);
  //psi_prime_scalarFT.initialize(latFT,1);
  //PlanFFT<Cplx> psi_prime_plan(&psi_prime, &psi_prime_scalarFT);
  short_wave.initialize(lat,1);
  short_wave_scalarFT.initialize(latFT,1);
  PlanFFT<Cplx> short_wave_plan(&short_wave, &short_wave_scalarFT);
  relativistic_term.initialize(lat,1);
  relativistic_term_scalarFT.initialize(latFT,1);
  PlanFFT<Cplx> relativistic_term_plan(&relativistic_term, &relativistic_term_scalarFT);
  stress_tensor.initialize(lat,1);
  stress_tensor_scalarFT.initialize(latFT,1);
  PlanFFT<Cplx> stress_tensor_plan(&stress_tensor, &stress_tensor_scalarFT);
  #endif
	//pi_k kessence
	pi_k.initialize(lat,1);
	scalarFT_pi.initialize(latFT,1);
	PlanFFT<Cplx> plan_pi_k(&pi_k, &scalarFT_pi);
	//zeta_integer_k kessence
	// zeta_half.initialize(lat,1);
	// scalarFT_zeta_half.initialize(latFT,1);
	// PlanFFT<Cplx> plan_zeta_half(&zeta_half, &scalarFT_zeta_half);
  //zeta_half_k kessence
  zeta_half.initialize(lat,1);
  scalarFT_zeta_half.initialize(latFT,1);
  PlanFFT<Cplx> plan_zeta_half(&zeta_half, &scalarFT_zeta_half);
	//chi_old initialize
	chi_old.initialize(lat,1);
	scalarFT_chi_old.initialize(latFT,1);
	PlanFFT<Cplx> plan_chi_old(&chi_old, &scalarFT_chi_old);
	//Stress tensor initializing
	T00_Kess.initialize(lat,1);
	T00_KessFT.initialize(latFT,1);
	PlanFFT<Cplx> plan_T00_Kess(&T00_Kess, &T00_KessFT);
	// T00_Kess.alloc();  // It seems we don't need it!
	T0i_Kess.initialize(lat,3);
	T0i_KessFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_T0i_Kess(&T0i_Kess, &T0i_KessFT);
	// T0i_Kess.alloc();
	Tij_Kess.initialize(lat,3,3,symmetric);
	Tij_KessFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_Tij_Kess(&Tij_Kess, &Tij_KessFT);
	// Tij_Kess.alloc();
	// kessence end


	update_cdm_fields[0] = &phi;
	update_cdm_fields[1] = &chi;
	update_cdm_fields[2] = &Bi;

	update_b_fields[0] = &phi;
	update_b_fields[1] = &chi;
	update_b_fields[2] = &Bi;

	update_ncdm_fields[0] = &phi;
	update_ncdm_fields[1] = &chi;
	update_ncdm_fields[2] = &Bi;

	Site x(lat);
	rKSite kFT(latFT);

	dx = 1.0 / (double) sim.numpts;
	numpts3d = (long) sim.numpts * (long) sim.numpts * (long) sim.numpts;
	

	for (i = 0; i < 3; i++) // particles may never move farther than to the adjacent domain
	{
		if (lat.sizeLocal(i)-1 < sim.movelimit)
			sim.movelimit = lat.sizeLocal(i)-1;
	}
	parallel.min(sim.movelimit);
	fourpiG = 1.5 * sim.boxsize * sim.boxsize / C_SPEED_OF_LIGHT / C_SPEED_OF_LIGHT; // Just a definition to make Friedmann equation simplified! and working with normal numbers
   COUT<<"Gevolution H0: "<<sqrt(2. * fourpiG / 3.)<<endl;
   COUT<<"Box: "<<sim.boxsize<<endl;
   COUT << "CLASS H0: " << gsl_spline_eval(H_spline, 1., acc) << endl;
	a = 1. / (1. + sim.z_in);
  tau = particleHorizon(a, fourpiG,
    #ifdef HAVE_HICLASS_BG
    gsl_spline_eval(H_spline, 1., acc), class_background
    #else
    cosmo
    #endif
  );

  if (sim.Cf * dx < sim.steplimit / Hconf(a, fourpiG,
    #ifdef HAVE_HICLASS_BG
      H_spline, acc
    #else
      cosmo
    #endif
  ) )
    // dtau = sim.Cf * dx / cosmo.n_mg_numsteps;
    dtau = sim.Cf * dx;
  else
    dtau = sim.steplimit / 	Hconf(a, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc
      #else
        cosmo
      #endif
      );

	dtau_old = 0.;
	if (ic.generator == ICGEN_BASIC)
		generateIC_basic(sim, ic, cosmo, fourpiG, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &pi_k, &zeta_half, &chi, &Bi, &source, &Sij, &scalarFT, &scalarFT_pi, &scalarFT_zeta_half, &BiFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, params, numparam);
	// generates ICs on the fly
	else if (ic.generator == ICGEN_READ_FROM_DISK)
  readIC(sim, ic, cosmo, fourpiG, a, tau, dtau, dtau_old, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, cycle, snapcount, pkcount, restartcount, IDbacklog, params, numparam);
#ifdef ICGEN_PREVOLUTION
	else if (ic.generator == ICGEN_PREVOLUTION)
		generateIC_prevolution(sim, ic, cosmo, fourpiG, a, tau, dtau, dtau_old, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, params, numparam);
#endif
#ifdef ICGEN_FALCONIC
	else if (ic.generator == ICGEN_FALCONIC)
		maxvel[0] = generateIC_FalconIC(sim, ic, cosmo, fourpiG, dtau, &pcls_cdm, pcls_ncdm, maxvel+1, &phi, &source, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_source, &plan_chi, &plan_Bi, &plan_source, &plan_Sij);
#endif
	else
	{
		COUT << " error: IC generator not implemented!" << endl;
		parallel.abortForce();
	}

	if (sim.baryon_flag > 1)
	{
		COUT << " error: baryon_flag > 1 after IC generation, something went wrong in IC generator!" << endl;
		parallel.abortForce();
	}

	numspecies = 1 + sim.baryon_flag + cosmo.num_ncdm;
	parallel.max<double>(maxvel, numspecies);

	if (sim.gr_flag > 0)
	{
		for (i = 0; i < numspecies; i++)
			maxvel[i] /= sqrt(maxvel[i] * maxvel[i] + 1.0);
	}

#ifdef CHECK_B
	if (sim.vector_flag == VECTOR_ELLIPTIC)
	{
		for (kFT.first(); kFT.test(); kFT.next())
		{
			BiFT_check(kFT, 0) = BiFT(kFT, 0);
			BiFT_check(kFT, 1) = BiFT(kFT, 1);
			BiFT_check(kFT, 2) = BiFT(kFT, 2);
		}
	}
#endif
#ifdef VELOCITY
	a_old = a;
	projection_init(&vi);
#endif

#ifdef BENCHMARK
	initialization_time = MPI_Wtime() - start_time;
	parallel.sum(initialization_time);
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << " BENCHMARK: " << hourMinSec(initialization_time) << endl << endl;
#else
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << endl << endl;
#endif

#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)

  if(sim.fluid_flag > 0 )
  {
    if(parallel.isRoot())  cout << " \033[1;31merror:\033[0m"<< " \033[1;31merror: You are using k-evolution and asking for fluid k-essence treatment at the same time! Don't know what to do!  \033[0m" << endl;
    parallel.abortForce();
  }
	if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
	{
		initializeCLASSstructures(sim, ic, cosmo, class_background, class_thermo, class_perturbs, params, numparam);
		if (sim.gr_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.) && (ic.generator == ICGEN_BASIC || (ic.generator == ICGEN_READ_FROM_DISK && cycle == 0)))
		{
			prepareFTchiLinear(class_background, class_perturbs, scalarFT, sim, ic, cosmo, fourpiG, a);
			plan_source.execute(FFT_BACKWARD);
			for (x.first(); x.test(); x.next())
				chi(x) += source(x);
			chi.updateHalo();
		}
	}
	if (numparam > 0) free(params);
#endif


#ifdef NONLINEAR_TEST
#ifdef 	FLUID_VARIABLES
//COUT << "H_spline = " << gsl_spline_eval(H_spline, 1., acc) << endl;
//COUT << "H_conf = " << Hconf(1., fourpiG,
//		  #ifdef HAVE_HICLASS_BG
//	        H_spline, acc
//	      #else
//            cosmo
//	      #endif 
//		  ) << endl;
// In case we want to initialize the IC ourselves
// the fluid variables are completely determined by the fields
for (x.first(); x.test(); x.next())
  {
    zeta_half(x)                = 0.0;
    zeta_half_old(x)            = 0.0;
	zeta_integer(x)             = 0.0;
    pi_k(x)                     = 0.0;

	phi_old(x)                  = 0.0;
	chi_old(x)                  = 0.0;
	
	// setting everything to zero just to be sure that there is a numer associated with every lattice point at all times
	delta_rho_fluid(x)          = 0.0;
	delta_p_fluid(x)            = 0.0;
	v_upper_i_fluid(x,0)        = 0.0;
	v_upper_i_fluid(x,1)        = 0.0;
	v_upper_i_fluid(x,2)        = 0.0;
	Sigma_upper_ij_fluid(x,0,0) = 0.0;
	Sigma_upper_ij_fluid(x,1,1) = 0.0;
	Sigma_upper_ij_fluid(x,2,2) = 0.0;
	Sigma_upper_ij_fluid(x,0,1) = 0.0;
	Sigma_upper_ij_fluid(x,0,2) = 0.0;
	Sigma_upper_ij_fluid(x,1,2) = 0.0;
	div_v_upper_fluid(x)        = 0.0; 
	phi_prime(x)                = 0.0;
	//v_x_fluid(x) = 0.0;
	//v_y_fluid(x) = 0.0;
	//v_z_fluid(x) = 0.0;
  }
  zeta_integer.updateHalo();
  zeta_half.updateHalo();  // communicate halo values
  pi_k.updateHalo();  // communicate halo values
  phi_prime.updateHalo();
  //zeta_half_old.updateHalo();
  //delta_rho_fluid.updateHalo();
  //delta_p_fluid.updateHalo();
  //v_upper_i_fluid.updateHalo();
  //Sigma_upper_ij_fluid.updateHalo();
  //div_v_upper_fluid.updateHalo();
 
////  Site mySite(lat);
////  std::string output_path_test = sim.output_path;
////  double i_d, j_d, k_d;
////  for (int i=0;i<sim.numpts; i++){
////  	for (int j=0;j<sim.numpts; j++){
////  		for (int k=0;k<sim.numpts; k++){
////			i_d = static_cast<double>(i);
////			j_d = static_cast<double>(j);
////			k_d = static_cast<double>(k);
////			if(mySite.setCoord(i,j,k)) v_upper_i_fluid(mySite,0) = 1.;   //pow(i_d*i_d + j_d*j_d + k_d*k_d,1./2.);
////		}
////	}
////  }
////  //v_upper_i_fluid.updateHalo();
////  v_upper_i_fluid.saveHDF5(output_path_test + "velocity_test.h5");
////  for (x.first(); x.test(); x.next())
////  {
////	//COUT <<  << endl;
////	//delta_rho_fluid(x) = 0.0;
////	delta_p_fluid(x) = (delta_rho_fluid(x + 0) - delta_rho_fluid(x - 0))/(2. * dx);
////
////  }
////  delta_rho_fluid.updateHalo();
////  delta_p_fluid.updateHalo();
////  delta_p_fluid.saveHDF5(output_path_test + "derivative_x.h5");
#endif

//   //****************************
//   //****SAVE DATA To test Backreaction
//   //****************************
  std::ofstream rho_i_rho_crit_0;
  //std::ofstream convert_to_cosmic_velocity;	
  std::ofstream Result_avg;
  std::ofstream Result_real;
  std::ofstream Result_fourier;
  std::ofstream Result_max;
  std::ofstream Redshifts;
  std::ofstream kess_snapshots;
  std::ofstream div_variables;
  std::ofstream potentials;
  std::ofstream Omega;
  std::string output_path = sim.output_path;
  //std::string filename_convert_to_cosmic_velocity = output_path + "convert_to_cosmic_velocity.txt";
  std::string filename_rho_i_rho_crit_0 = output_path + "rho_i_rho_crit_0.txt";
  std::string filename_avg = output_path + "Result_avg.txt";
  std::string filename_real = output_path + "Result_real.txt";
  std::string filename_fourier = output_path + "Result_fourier.txt";
  std::string filename_max = output_path + "Results_max.txt";
  std::string filename_redshift = output_path + "redshifts.txt";
  std::string filename_kess_snapshots = output_path + "kess_snapshots.txt";
  std::string filename_div_variables = output_path + "div_variables.txt";
  std::string filename_potentials = output_path + "potentials.txt";
  std::string filename_Omega = output_path + "Omega.txt";

  rho_i_rho_crit_0.open(filename_rho_i_rho_crit_0, std::ios::out);
  //convert_to_cosmic_velocity.open(filename_convert_to_cosmic_velocity, std::ios::out);
  Result_avg.open(filename_avg, std::ios::out);
  Result_real.open(filename_real, std::ios::out);
  Result_fourier.open(filename_fourier, std::ios::out);
  Result_max.open(filename_max, std::ios::out);
  Redshifts.open(filename_redshift, std::ios::out);
  kess_snapshots.open(filename_kess_snapshots, std::ios::out);
  div_variables.open(filename_div_variables, std::ios::out);
  potentials.open(filename_potentials, std::ios::out);
  Omega.open(filename_Omega, std::ios::out);


  //convert_to_cosmic_velocity << "###    type(gev/kess)[0],     snapcount[1],         a[2],              z[3],            Delta conformal time / Delta cosmic time[4]" << endl;

  Omega << "###  scale factor [0], redshift [1],      Omega_DE [2],    Omega_CDM [3],    Omega_baryons [4],    Omega_photons [5],      Omega_massless_neutrinos [6]" << endl;

  rho_i_rho_crit_0 << "### a [0],    z [1],              rho_DE/rho_crit_0 [2],        rho_M/rho_crit_0 [3],       rho_Rad/rho_crit_0 [4]" << endl;

  div_variables<<"### Here are the variables" << endl;
  div_variables<<"cs2_kessence         " << gsl_spline_eval(cs2_spline, a, acc) << endl;
  div_variables<<"N_kessence           " << sim.nKe_numsteps << endl;
  div_variables<<"### (z),       H_conf*avg_pi,       max |H_conf*pi_k|,       avg_zeta,      max_abs_zeta"  <<endl;

  potentials << "### z,       relative change in Phi,           relative change in Psi" <<endl;
	

  Result_avg<<"### The result of the average over time \n### d tau = "<< dtau<<endl;
  Result_avg<<"### number of kessence update = "<<  sim.nKe_numsteps <<endl;
  Result_avg<<"### initial time = "<< tau <<endl;
  Result_avg<<"### 1- tau\t2- average(H pi_k)\t3- average (zeta)\t 4- average (phi)\t5-z(redshift)   " <<endl;


  Result_max<<"### The result of the maximum over time \n### d tau = "<< dtau<<endl;
  Result_max<<"### number of kessence update = "<<  sim.nKe_numsteps <<endl;
  Result_max<<"### initial time = "<< tau <<endl;
  Result_max<<"### 1- tau\t2- max(H pi_k)\t3- max (zeta)\t 4- max (phi)   " <<endl;


  Result_real<<"### The result of the verage over time \n### d tau = "<< dtau<<endl;
  Result_real<<"### number of kessence update = "<<  sim.nKe_numsteps <<endl;
  Result_real<<"### initial time = "<< tau <<endl;
  Result_real<<"### 1- tau\t2- pi_k(x)\t3-zeta(x)\t 4-x" <<endl;


  Result_fourier<<"### The result of the average over time \n### d tau = "<< dtau<<endl;
  Result_fourier<<"### number of kessence update = "<<  sim.nKe_numsteps <<endl;
  Result_fourier<<"### initial time = "<< tau <<endl;
  Result_fourier<<"### 1- tau\t 2- pi_k(k)\t\t3-zeta(k)\t\t4-|k|\t\t 5-vec{k} \t 6-|k|^2"<<endl;


  kess_snapshots<<"### The result of the snapshots produced over time for blow-up \n### d tau = "<< dtau<<endl;
  kess_snapshots<<"### number of kessence update = "<<  sim.nKe_numsteps <<endl;
  kess_snapshots<<"### initial time = "<< tau <<endl;
  kess_snapshots<<"### H0 = " <<Hconf(1., fourpiG,
		  #ifdef HAVE_HICLASS_BG
	        H_spline, acc
	      #else
            cosmo
	      #endif 
		  )  << endl;
  kess_snapshots<<"### 1- tau\t2- z \t3- a\t 4- zeta_avg\t 5- avg_pi\t 6- avg_phi\t 7- tau/boxsize\t 8- H_conf/H0 \t 9- snap_count"<<endl;


//defining the average
double avg_pi = 0.;
double avg_zeta = 0.;
double avg_phi = 0.;
double max_zeta_old = 0.;
double avg_zeta_old = 0.;
double avg_pi_old = 0.;

double max_pi = 0.;
double max_zeta = 0.;
double max_phi = 0.;

int norm_kFT_squared = 0.;

// HDF5 outputs!
string str_filename ;
string str_filename2 ;
string str_filename3 ;

//double previous_a_kess;
//int numpts = sim.numpts;
//double previous_avg_pi;
//double previous_avg_zeta;
double max_abs_pi;
double max_abs_zeta;
//double previous_max_abs_zeta;
//double previous_max_abs_pi;
double energy_overdensity_Kess;
double alternative_energy_overdensity_Kess;

std::vector<double> Omega_vector;
double Gevolution_H0 = sqrt(2. * fourpiG / 3.);
div_variables << "### Gevolution_H0 " <<  Gevolution_H0 << endl;
double a_old_for_kess_velocity;
#endif

	//******************************************************************
	//Write spectra check!
	// Kessence projection Tmunu Test IC
	//******************************************************************
	//  	if (sim.vector_flag == VECTOR_ELLIPTIC)
	// 		{
	// 			projection_Tmunu_kessence( T00_Kess,T0i_Kess,Tij_Kess, dx, a, phi, phi_old, chi, pi_k, zeta_integer_k, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hconf(a, fourpiG, cosmo), fourpiG, 1 );
	// 		}
	//  	else
	// 		{
	// 			projection_Tmunu_kessence( T00_Kess,T0i_Kess,Tij_Kess, dx, a, phi, phi_old, chi, pi_k, zeta_integer_k, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hconf(a, fourpiG, cosmo), fourpiG, 0 );
	// 		}
	//
// writeSpectra(sim, cosmo, fourpiG, a, pkcount, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k, &zeta_half, &chi, &Bi, &T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT ,&scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij);

// writeSpectra_phi_prime(sim, cosmo, fourpiG, a, pkcount, &phi_prime, &phi_prime_scalarFT, &phi_prime_plan);

// writeSpectra(sim, cosmo, fourpiG, a, pkcount, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k, &zeta_half, &chi, &Bi, &T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT ,&scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij);
	
	// Calculating the dimensionless energy desities, i.e., Omega_i(a) = rho_i(a)/rho_crit(a). This is done using hiclass.
	#ifdef NONLINEAR_TEST

//	//double proper_time = proper_time_gev(gsl_spline_eval(time_spline, a_for_proper_time, acc),gsl_spline_eval(H_spline,1.,acc),fourpiG); // proper time in gevolution units
//    //double proper_time_test = gsl_spline_eval(time_spline, a_for_proper_time, acc)*299792458.*60.*60.*24.*365.*10**9./3.086e+22 * gsl_spline_eval(H_spline,1.,acc)/sqrt(2./3.*fourpiG); // proper time in gevolution units
//	double dt = proper_time_gev(gsl_spline_eval(time_spline, a_for_proper_time, acc),gsl_spline_eval(H_spline,1.,acc),fourpiG) - proper_time_gev(gsl_spline_eval(time_spline, a, acc),gsl_spline_eval(H_spline,1.,acc),fourpiG);
	//double dtau_dt = dtau/dt;
	//COUT << "dtau_dt    "  <<  dtau_dt << endl;


	//double tau_for_testing = particleHorizon(0.5, fourpiG,
    //#ifdef HAVE_HICLASS_BG
    //gsl_spline_eval(H_spline, 1., acc), class_background
    //#else
    //cosmo
    //#endif
    //);
  
  
  //double tau_from_class = gsl_spline_eval(conformal_time_spline, 0.5, acc)*gsl_spline_eval(H_spline,1.,acc)/sqrt(2./3.*fourpiG);
  //COUT   <<"testing,    spline      " <<tau_for_testing << "    "  << tau_from_class << endl;
  //double proper_time_test = gsl_spline_eval(time_spline, 0.5, acc)*299792458.*60.*60.*24.*365.*10**9./3.086e+22 * gsl_spline_eval(H_spline,1.,acc)/sqrt(2./3.*fourpiG); // proper time in gevolution units


	// Writing the time evolution of Omega_i to file. a is spaced as a = np.logspace(-15,1,1000,base=np.exp(1))
	for (int k = 0; k < 1000; k++){
		const double c_1 = exp(-15.);
        const double c_2 = 1./999.*log(exp(1.)/c_1);
		double a_for_plotting = c_1*exp(k*c_2);  // making an array with logarithmic spacing, i.e., x_i = c_1*e^(i*c_2).
		Omega_vector = calculate_Omega(gsl_spline_eval(rho_crit_spline, a_for_plotting, acc),gsl_spline_eval(rho_smg_spline, a_for_plotting, acc),gsl_spline_eval(rho_cdm_spline, a_for_plotting, acc),gsl_spline_eval(rho_b_spline, a_for_plotting, acc),gsl_spline_eval(rho_g_spline, a_for_plotting, acc),gsl_spline_eval(rho_ur_spline, a_for_plotting, acc));
		Omega << a_for_plotting << "     " << 1./a_for_plotting - 1. << "     " << Omega_vector[0] << "     " << Omega_vector[1] << "     " << Omega_vector[2] << "        "   << Omega_vector[3]<< "        "   << Omega_vector[4] << endl;
		rho_i_rho_crit_0 << a_for_plotting << "     " << 1./a_for_plotting - 1.  << "    " << gsl_spline_eval(rho_smg_spline, a_for_plotting, acc)/gsl_spline_eval(rho_crit_spline, 1., acc)<< "     " <<  (gsl_spline_eval(rho_b_spline, a_for_plotting, acc) + gsl_spline_eval(rho_cdm_spline, a_for_plotting, acc))/gsl_spline_eval(rho_crit_spline, 1., acc) <<"       " << (gsl_spline_eval(rho_g_spline, a_for_plotting, acc) + gsl_spline_eval(rho_ur_spline, a_for_plotting, acc))/gsl_spline_eval(rho_crit_spline, 1., acc) << endl;
	}
	Omega.close();
	#endif
	//double age_test = gsl_spline_eval(time_spline, 1., acc)*3.086e+22/299792458.*1./(60.*60.*24.*365.*pow(10.,9.));
	//double age_test = gsl_spline_eval(time_spline, 0.7731939999999999, acc);
	//cout <<"age of universe is " <<"       " << age_test<< endl; // cosmic time in Gyr
	//parallel.abortForce();
	while (true)    // main loop
	{
		//avg_T00_Kess_file << a << "     " << 1./a - 1.  << "    " << gsl_spline_eval(rho_smg_spline, a, acc)/gsl_spline_eval(rho_crit_spline, 1., acc)<< "     " <<  (gsl_spline_eval(rho_b_spline, a, acc) + gsl_spline_eval(rho_cdm_spline, a, acc))/gsl_spline_eval(rho_crit_spline, 1., acc) << endl;
		//#ifdef NONLINEAR_TEST
		// "###  scale factor [0], conformal time [1], redshift [2],      Omega_DE [3],    Omega_CDM [4],    Omega_baryons [5],    Omega_radiation [6]"
		
		//for (int iteration_in_a = 0; iteration_in_a < 1000; iteration_in_a++){

		//} 
		//Omega_vector = calculate_Omega(gsl_spline_eval(rho_crit_spline, a, acc),gsl_spline_eval(rho_smg_spline, a, acc),gsl_spline_eval(rho_cdm_spline, a, acc),gsl_spline_eval(rho_b_spline, a, acc),gsl_spline_eval(rho_g_spline, a, acc));
		//Omega << a << "     " << tau << "     " << 1./a - 1. << "     " << Omega_vector[0] << "     " << Omega_vector[1] << "     " << Omega_vector[2] << "        "   << Omega_vector[3] << endl;
		//#endif 
		
		// checking if potentials change much
		// phi and chi are here the initial conditions at cycle 0
		if (cycle > 1){
			double temp_relative_Phi;
			double Psi;
			double Psi_old;
			double temp_relative_Psi;
			double relative_Phi = 0.0;
			double relative_Psi = 0.0;
			double temp_max_Phi;
			double temp_max_Psi;
			double max_Phi = 0.0;
			double max_Psi = 0.0;
			double avg_rel_Phi = 0.0;


			for (x.first(); x.test(); x.next()){
				temp_relative_Phi = (phi(x)-phi_old(x))/phi_old(x);
				//temp_relative_Phi = phi_prime(x);
				Psi = phi(x) - chi(x);
				Psi_old = phi_old(x) - chi_old(x);
				temp_relative_Psi = (Psi-Psi_old)/Psi_old;
				if (temp_relative_Phi < 0.0) temp_relative_Phi *= -1.;
				avg_rel_Phi += temp_relative_Phi;
				if (temp_relative_Psi < 0.0) temp_relative_Psi *= -1.;
				if (phi(x)<0.0) temp_max_Phi = -phi(x);
					else temp_max_Phi = phi(x);
				if (temp_max_Phi > max_Phi) max_Phi = temp_max_Phi;
				if (Psi<0.0) temp_max_Psi = -Psi;
					else temp_max_Psi = Psi;
				if (temp_max_Psi > max_Psi) max_Psi = temp_max_Psi;
				if (temp_relative_Phi > relative_Phi) relative_Phi = temp_relative_Phi;
				if (temp_relative_Psi > relative_Psi) relative_Psi = temp_relative_Psi;
			}
			
			// summing over all processes
			parallel.sum(avg_rel_Phi);
			avg_rel_Phi /= numpts3d;

			// finding the maximum over all processes
			parallel.max(relative_Phi);
			parallel.max(relative_Psi);
			parallel.max(max_Phi);
			parallel.max(max_Psi);

			////if (relative_Phi > 0.01) COUT <<COLORTEXT_RED <<"Warning: relative change in Phi is " <<COLORTEXT_RESET<<std::fixed<<std::setprecision(2) <<100.0*relative_Phi<<"%"<< endl;
			////if (relative_Psi > 0.01) COUT <<COLORTEXT_RED <<"Warning: relative change in Psi is " <<COLORTEXT_RESET<<std::fixed<<std::setprecision(2) <<100.0*relative_Psi<<"%"<< endl;

			potentials << 1./a -1. <<"     "<<relative_Phi<<"     "<<relative_Psi<< "		" <<  max_Phi << "		"<<max_Psi << "        "<< avg_rel_Phi  <<endl; 
		}
	if (dtau_old>0.0){
  	for (x.first(); x.test(); x.next())
  		{
  			phi_old(x) = phi(x);
  			chi_old(x) = chi(x);
			//if (cycle>0) update_zeta_eq()
			
         // if(x.coord(0)==32 && x.coord(1)==12 && x.coord(2)==32) cout<<"zeta_half: "<<zeta_half(x)<<endl;
  		}
	}
	
		//if (cycle==0)phi_old.saveHDF5(output_path + "phi_old_test.h5");
		//parallel.abortForce();

		
#ifdef NONLINEAR_TEST
	phi_old.updateHalo();
	chi_old.updateHalo();
#endif 
//COUT << 791 << endl;
#ifdef NONLINEAR_TEST
//COUT <<"rho_crit = " << gsl_spline_eval(rho_crit_spline, 1., acc) << endl;
      //****************************
      //****PRINTING AVERAGE OVER TIME
      //****************************
      // check_field(  zeta_half, 1. , " H pi_k", numpts3d);
	
      avg_pi =average(  pi_k, Hconf(a, fourpiG,
	  #ifdef HAVE_HICLASS_BG
	  H_spline,acc
	  #else
	  cosmo
	  #endif
	  ), numpts3d ) ;    
      avg_zeta =average(  zeta_half,1., numpts3d ) ;
      avg_phi =average(  phi , 1., numpts3d ) ;

      max_pi =maximum(  pi_k,
        Hconf(a, fourpiG,
        #ifdef HAVE_HICLASS_BG
          H_spline, acc
        #else
          cosmo
        #endif
	  ),numpts3d ) ;
      max_zeta =maximum(  zeta_half,1., numpts3d ) ;
      max_phi =maximum(  phi , 1., numpts3d ) ;

      COUT << scientific << setprecision(8);
      // if(parallel.isRoot())
      // {
        // fprintf(Result_avg,"\n %20.20e %20.20e ", tau, avg ) ;
      Result_avg<<setw(9) << tau <<"\t"<< setw(9) << avg_pi<<"\t"<< setw(9) << avg_zeta<<"\t"<< setw(9) << avg_phi<<"\t"<< setw(9) << 1./a -1. << endl;

        Result_max<<setw(9) << tau <<"\t"<< setw(9) << max_pi<<"\t"<< setw(9) << max_zeta<<"\t"<< setw(9) << max_phi<<endl;

      // }
      //****************************
      //****PRINTING REAL SPACE INFO
      //****************************
      for (x.first(); x.test(); x.next())
    	{
          //NL_test, Printing out average
        if(x.coord(0)==32 && x.coord(1)==20 && x.coord(2)==10)
        {
          // if(parallel.isRoot())
          // {
          Result_real<<setw(9) << tau <<"\t"<< setw(9) <<pi_k (x)<<"\t"<< setw(9)<<zeta_half (x)<<"\t"<<x<<endl;
          // }
        }
    	}
      //****************************
      //FOURIER PRINTING
      //****************************
      for(kFT.first();kFT.test();kFT.next())
      {
        norm_kFT_squared= kFT.coord(0)*kFT.coord(0) + kFT.coord(1) * kFT.coord(1) + kFT.coord(2) * kFT.coord(2);
        if(norm_kFT_squared == 1)
        {
          Result_fourier<<setw(9) << tau <<"\t"<< setw(9) << scalarFT_pi(kFT)<<"\t"<< setw(9)<<scalarFT_zeta_half (kFT)<<"\t"<<kFT<<"\t"<<norm_kFT_squared<<endl;
        }
      }
      //**********************
      //END ADDED************
      //**********************
#endif

#ifdef BENCHMARK
		cycle_start_time = MPI_Wtime();
#endif
		// construct stress-energy tensor
		projection_init(&source);
#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
		if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
			projection_T00_project(class_background, class_perturbs, source, scalarFT, &plan_source, sim, ic, cosmo, fourpiG, a);
#endif
		if (sim.gr_flag > 0)
		{
			projection_T00_project(&pcls_cdm, &source, a, &phi);




			if (sim.baryon_flag)
				projection_T00_project(&pcls_b, &source, a, &phi);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T00_project(pcls_ncdm+i, &source, a, &phi);
				else if (sim.radiation_flag == 0 || (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] == 0))
				{
					tmp = bg_ncdm(a, cosmo, i);
					for(x.first(); x.test(); x.next())
						source(x) += tmp;
				}
			}
		}
		else
		{
			scalarProjectionCIC_project(&pcls_cdm, &source);
			if (sim.baryon_flag)
				scalarProjectionCIC_project(&pcls_b, &source);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					scalarProjectionCIC_project(pcls_ncdm+i, &source);
			}
		}
		projection_T00_comm(&source);

#ifdef VELOCITY
		if ((sim.out_pk & MASK_VEL) || (sim.out_snapshot & MASK_VEL))
		{
			projection_init(&Bi);
            projection_Ti0_project(&pcls_cdm, &Bi, &phi, &chi);
            vertexProjectionCIC_comm(&Bi);
            compute_vi_rescaled(cosmo, &vi, &source, &Bi, a, a_old
              #ifdef HAVE_HICLASS_BG
              , H_spline, acc
              #endif
            );
            a_old = a;
		}
#endif

		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
			projection_init(&Bi);
			projection_T0i_project(&pcls_cdm, &Bi, &phi);
			if (sim.baryon_flag)
				projection_T0i_project(&pcls_b, &Bi, &phi);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_Bncdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T0i_project(pcls_ncdm+i, &Bi, &phi);
			}
			projection_T0i_comm(&Bi);
		}

		projection_init(&Sij);
		projection_Tij_project(&pcls_cdm, &Sij, a, &phi);
		if (sim.baryon_flag)
			projection_Tij_project(&pcls_b, &Sij, a, &phi);
		if (a >= 1. / (sim.z_switch_linearchi + 1.))
		{
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_Tij_project(pcls_ncdm+i, &Sij, a, &phi);
			}
		}
		projection_Tij_comm(&Sij);

#ifdef BENCHMARK
		projection_time += MPI_Wtime() - cycle_start_time;
		ref_time = MPI_Wtime();
#endif

if (sim.Kess_source_gravity==1)
{
  Hc = Hconf(a, fourpiG,
  #ifdef HAVE_HICLASS_BG
    H_spline, acc
  #else
    cosmo
  #endif
    ); 
// Kessence projection Tmunu
// In the projection zeta_integer comes, since synched with particles..

#ifdef HAVE_HICLASS_BG // hiclass used to provide quantities!

if (sim.vector_flag == VECTOR_ELLIPTIC)
  {
    projection_Tmunu_kessence_eq(T00_Kess, T0i_Kess, Tij_Kess, dx, a, phi, phi_old, chi, pi_k, zeta_half, gsl_spline_eval(rho_smg_spline, a, acc), gsl_spline_eval(p_smg_spline, a, acc), gsl_spline_eval(rho_crit_spline, 1., acc), gsl_spline_eval(p_smg_prime_spline, a, acc)/gsl_spline_eval(rho_smg_prime_spline, a, acc), gsl_spline_eval(cs2_spline, a, acc), Hc, sim.NL_kessence, 1);
  }
else
  {
    projection_Tmunu_kessence_eq(T00_Kess, T0i_Kess, Tij_Kess, dx, a, phi, phi_old, chi, pi_k, zeta_integer, gsl_spline_eval(rho_smg_spline, a, acc), gsl_spline_eval(p_smg_spline, a, acc), gsl_spline_eval(rho_crit_spline, 1., acc), gsl_spline_eval(p_smg_prime_spline, a, acc)/gsl_spline_eval(rho_smg_prime_spline, a, acc), gsl_spline_eval(cs2_spline, a, acc), Hc, sim.NL_kessence, 0);
  }
#else // default kevolution // No hiclass BG used
 	if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
			projection_Tmunu_kessence(T00_Kess, T0i_Kess,Tij_Kess, dx, a, phi, phi_old, 	chi, pi_k, zeta_half, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hc, fourpiG, sim.NL_kessence, 1);
		}
 	else
		{
			projection_Tmunu_kessence(T00_Kess, T0i_Kess,Tij_Kess, dx, a, phi, phi_old, 	chi, pi_k, zeta_half, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hc, fourpiG, sim.NL_kessence, 0);
		}
#endif

		for (x.first(); x.test(); x.next())
		{
       // if(x.coord(0)==32 && x.coord(1)==12 && x.coord(2)==32)cout<<"T00: "<< T00_Kess(x)<<" Phi:"<<phi(x)<<endl;
			// The coefficient is because it wanted to to be source according to eq C.2 of Gevolution paper
			// Note that it is multiplied to dx^2 and is divived by -a^3 because of definition of T00 which is scaled by a^3
			// We have T00 and Tij according to code's units, but source is important to calculate potentials and moving particles.
			// There is coefficient between Tij and Sij as source.
			source(x) += T00_Kess(x);
			if (sim.vector_flag == VECTOR_ELLIPTIC)for(int 	c=0;c<3;c++)Bi(x,c)+=  T0i_Kess(x,c);
			for(int c=0;c<6;c++)Sij(x,c)+=(2.) * Tij_Kess(x,c);
      // if(x.coord(0)==32 && x.coord(1)==20 && x.coord(2)==10)
      // {
      // cout<<"x"<<x<<"T00_Kess(x): "<<T00_Kess(x)<<endl;
      // }
		}
} 
#ifdef BENCHMARK
		kessence_update_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif
// Kessence projection Tmunu end

		if (sim.gr_flag > 0)
		{
			T00hom = 0.;
			for (x.first(); x.test(); x.next())
				T00hom += source(x);
			parallel.sum<double>(T00hom);
			T00hom /= (double) numpts3d;

			if (cycle % CYCLE_INFO_INTERVAL == 0)
			{
				COUT << " cycle " << cycle << ", background information: z = " << (1./a) - 1. << ", average T00 = " << T00hom << ", background model = " << cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo) << endl;
			}

			if (dtau_old > 0.)
			{
        Hc = Hconf(a, fourpiG,
        #ifdef HAVE_HICLASS_BG
          H_spline, acc
        #else
          cosmo
        #endif
          );
        #ifdef NONLINEAR_TEST
				prepareFTsource_BackReactionTest<Real>(short_wave, dx, phi, chi, source, cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo), source, 3. * Hc * dx * dx / dtau_old, fourpiG * dx * dx / a, 3. * Hc * Hc * dx * dx, sim.boxsize);  // prepare nonlinear source for phi update
        #else
        prepareFTsource<Real>(phi, chi, source, cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo), source, 3. * Hc * dx * dx / dtau_old, fourpiG * dx * dx / a, 3. * Hc * Hc * dx * dx);  // prepare nonlinear source for phi update
        #endif
#ifdef BENCHMARK
				ref2_time= MPI_Wtime();
#endif
				plan_source.execute(FFT_FORWARD);  // go to k-space
#ifdef BENCHMARK
				fft_time += MPI_Wtime() - ref2_time;
				fft_count++;
#endif

				solveModifiedPoissonFT(scalarFT, scalarFT, 1. / (dx * dx), 3. * Hc / dtau_old);  // phi update (k-space)



#ifdef BENCHMARK
				ref2_time= MPI_Wtime();
#endif
				plan_phi.execute(FFT_BACKWARD);	 // go back to position space
#ifdef BENCHMARK
				fft_time += MPI_Wtime() - ref2_time;
				fft_count++;
#endif
			}
		}
		else
		{
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			plan_source.execute(FFT_FORWARD);  // Newton: directly go to k-space
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count++;
#endif

			solveModifiedPoissonFT(scalarFT, scalarFT, fourpiG / a);  // Newton: phi update (k-space)

#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			plan_phi.execute(FFT_BACKWARD);	 // go back to position space
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count++;
#endif
		}

		phi.updateHalo();  // communicate halo values

		// record some background data
		if (kFT.setCoord(0, 0, 0))
		{
			sprintf(filename, "%s%s_background.dat", sim.output_path, sim.basename_generic);
			outfile = fopen(filename, "a");
			if (outfile == NULL)
			{
				cout << " error opening file for background output!" << endl;
			}
			else
			{
  #ifdef HAVE_HICLASS_BG 
        if (cycle == 0)
          fprintf(outfile, "# background statistics \n#All the values except 'a' and 'T00' are computed in the hiclass. the quantities rho_x, rho_x_prime are in unit of hiclass, T00 is in the k-evolution code's unit\n# 0:cycle             1:tau/boxsize             2:a             3:conf H/H0      4:conf H'/H0^2             5:alpha_K             6:c_s^2             7:s=(c_s^2)'/(H_conf c_s^2)             8:c_a^2=p_smg'/rho_smg'             9:rho_smg             10:rho_cdm      11:rho_b             12:rho_g             13:p_smg             14:rho_smg_prime             15:p_smg_prime             16:Omega_smg(rho_smg/rho_crit)             17:phi(k=0)             18:T00(k=0)\n");
        fprintf(outfile, " %6d   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e\n", cycle, tau, a, Hconf(a, fourpiG,H_spline, acc) /
        Hconf(1., fourpiG, H_spline, acc)
      , Hconf_prime(a, fourpiG, H_spline, acc)/Hconf(1., fourpiG, H_spline, acc)/Hconf(1., fourpiG, H_spline, acc)
      , gsl_spline_eval(alpha_K_spline, a, acc)
      , gsl_spline_eval(cs2_spline, a, acc)
      , gsl_spline_eval(cs2_prime_spline, a, acc)/gsl_spline_eval(cs2_spline, a, acc)/(a* gsl_spline_eval(H_spline, a, acc)) // s = (c_s^2)' [hiclass]/c_s^2/aH(hiclass) -- aH =ℋ, also prime means time derivative wrt conformal time.
      , gsl_spline_eval(p_smg_prime_spline, a, acc)/gsl_spline_eval(rho_smg_prime_spline, a, acc) //ca2=p_smg'/rho_smg'
      , gsl_spline_eval(rho_smg_spline, a, acc)
      , gsl_spline_eval(rho_cdm_spline, a, acc)
      , gsl_spline_eval(rho_b_spline, a, acc)
      , gsl_spline_eval(rho_g_spline, a, acc)
      , gsl_spline_eval(p_smg_spline, a, acc)
      , gsl_spline_eval(rho_smg_prime_spline, a, acc)
      , gsl_spline_eval(p_smg_prime_spline, a, acc)
      , gsl_spline_eval(rho_smg_spline, a, acc)/gsl_spline_eval(rho_crit_spline, a, acc)
      , scalarFT(kFT).real(), T00hom);
        fclose(outfile);
  #else
  if (cycle == 0)
    fprintf(outfile, "# background statistics\n# cycle   tau/boxsize    a             conformal H/H0             Hconf'/H0^2             phi(k=0)       T00(k=0)\n");
  fprintf(outfile, " %6d   %e   %e   %e   %e   %e   %e\n", cycle, tau, a, Hconf(a, fourpiG, cosmo) / Hconf(1., fourpiG, cosmo), Hconf_prime(a, fourpiG, cosmo) / Hconf(1., fourpiG, cosmo)/Hconf(1., fourpiG, cosmo), scalarFT(kFT).real(), T00hom);
  fclose(outfile);
  #endif
		  }
		}
		// done recording background data

		prepareFTsource<Real>(phi, Sij, Sij, 2. * fourpiG * dx * dx / a);  // prepare nonlinear source for additional equations
#ifdef BENCHMARK
		ref2_time= MPI_Wtime();
#endif
		plan_Sij.execute(FFT_FORWARD);  // go to k-space
#ifdef BENCHMARK
		fft_time += MPI_Wtime() - ref2_time;
		fft_count += 6;
#endif
#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
		if (sim.radiation_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.))
		{
			prepareFTchiLinear(class_background, class_perturbs, scalarFT, sim, ic, cosmo, fourpiG, a);
			projectFTscalar(SijFT, scalarFT, 1);
		}
		else
#endif
		projectFTscalar(SijFT, scalarFT);  // construct chi by scalar projection (k-space)
#ifdef BENCHMARK
		ref2_time= MPI_Wtime();
#endif
		plan_chi.execute(FFT_BACKWARD);	 // go back to position space
#ifdef BENCHMARK
		fft_time += MPI_Wtime() - ref2_time;
		fft_count++;
#endif
		chi.updateHalo();  // communicate halo values
		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			plan_Bi.execute(FFT_FORWARD);
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count++;
#endif
			projectFTvector(BiFT, BiFT, fourpiG * dx * dx); // solve B using elliptic constraint (k-space)
#ifdef CHECK_B
			evolveFTvector(SijFT, BiFT_check, a * a * dtau_old);
#endif
		}
		else
			evolveFTvector(SijFT, BiFT, a * a * dtau_old);  // evolve B using vector projection (k-space)
		if (sim.gr_flag > 0)
		{
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			plan_Bi.execute(FFT_BACKWARD);  // go back to position space
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count += 3;
#endif
			Bi.updateHalo();  // communicate halo values
		}

#ifdef BENCHMARK
		gravity_solver_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif
//COUT << 1205 << endl;

// lightcone output
if (sim.num_lightcone > 0)
  writeLightcones(sim, cosmo, fourpiG, a, tau, dtau, dtau_old, maxvel[0], cycle, h5filename + sim.basename_lightcone,
  #ifdef HAVE_HICLASS_BG
	class_background, H_spline, acc,
	#endif
  &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &Sij, &BiFT, &SijFT, &plan_Bi, &plan_Sij, done_hij, IDbacklog);
else done_hij = 0;

#ifdef BENCHMARK
lightcone_output_time += MPI_Wtime() - ref_time;
ref_time = MPI_Wtime();
#endif


if (dtau_old > 0.0){
	for (x.first(); x.test(); x.next())
	{
	  phi_prime(x) =(phi(x)-phi_old(x))/dtau_old;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#ifdef FLUID_VARIABLES
//  if (cycle > 1){
//	Hc = Hconf(a, fourpiG,
//  #ifdef HAVE_HICLASS_BG
//    H_spline, acc
//  #else
//    cosmo
//  #endif
//    ); 
//
//   calculate_fluid_properties(div_v_upper_fluid,Sigma_upper_ij_fluid, delta_rho_fluid,delta_p_fluid,v_upper_i_fluid,pi_k,zeta_integer,phi,chi,gsl_spline_eval(rho_smg_spline, a, acc)
//   	,gsl_spline_eval(p_smg_spline, a, acc),gsl_spline_eval(cs2_spline, a, acc), Hc,dx,a,gsl_spline_eval(rho_crit_spline, 1., acc), Gevolution_H0);
//	double max_drf = max_abs_func(delta_rho_fluid,1.);
//	COUT << max_drf << endl;
//  }
//	#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

if (snapcount < sim.num_snapshot && 1. / a < sim.z_snapshot[snapcount] + 1.)
{

  #ifdef FLUID_VARIABLES
  if (cycle > 1){ 
	Hc = Hconf(a, fourpiG,
  #ifdef HAVE_HICLASS_BG
    H_spline, acc
  #else
    cosmo
  #endif
    ); 

   calculate_fluid_properties(div_v_upper_fluid,Sigma_upper_ij_fluid, delta_rho_fluid,delta_p_fluid,v_upper_i_fluid,pi_k,zeta_integer,phi,chi,gsl_spline_eval(rho_smg_spline, a, acc)
   	,gsl_spline_eval(p_smg_spline, a, acc),gsl_spline_eval(cs2_spline, a, acc), Hc,dx,a,gsl_spline_eval(rho_crit_spline, 1., acc), Gevolution_H0);
    
	}
	else{
		COUT << COLORTEXT_RED << "Fluid properties at initial time are not well defined. They will be written to file anyway..." << COLORTEXT_RESET << endl;
	}
	#endif


	COUT << COLORTEXT_CYAN << " writing snapshot" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
	writeSnapshots(sim, cosmo, fourpiG, a, dtau_old, done_hij, snapcount, h5filename + sim.basename_snapshot,
          #ifdef HAVE_HICLASS_BG
          	H_spline, acc,
          #endif
          	&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k,&zeta_half, &chi, &Bi, &T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij
	#ifdef CHECK_B
		, &Bi_check, &BiFT_check, &plan_Bi_check
	#endif
    #ifdef VELOCITY
		, &vi
    #endif
    	#ifdef FLUID_VARIABLES
           ,&delta_rho_fluid, &delta_p_fluid, &v_upper_i_fluid, &Sigma_upper_ij_fluid, &div_v_upper_fluid
    #endif
	);
	snapcount++;
}

#ifdef BENCHMARK
		snapshot_output_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif

		// power spectra
		if (pkcount < sim.num_pk && 1. / a < sim.z_pk[pkcount] + 1.)
		{
			COUT << COLORTEXT_CYAN << " writing power spectra" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

#ifdef NONLINEAR_TEST
      writeSpectra_PoissonTerms(sim,  cosmo,  fourpiG,  a, pkcount,
      #ifdef HAVE_HICLASS_BG
      H_spline, acc,
      #endif
      &short_wave, &short_wave_scalarFT , &short_wave_plan);
      writeSpectra_phi_prime(sim, cosmo, fourpiG, a, pkcount,
      #ifdef HAVE_HICLASS_BG
      H_spline, acc,
      #endif
      &phi_prime, &phi_prime_scalarFT, &phi_prime_plan);
#endif

			writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
				class_background, class_perturbs, ic,
  #ifdef HAVE_HICLASS_BG
        H_spline, acc, gsl_spline_eval(rho_smg_spline, a, acc), gsl_spline_eval(rho_crit_spline, 1., acc),
  #endif
#endif
				&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k,&zeta_half, &chi, &Bi, &T00_Kess, &T0i_Kess, &Tij_Kess ,&source, &Sij, &scalarFT, &scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k , &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi, &viFT, &plan_vi
#endif
			);

			pkcount++;
		}

    #ifdef EXACT_OUTPUT_REDSHIFTS
    		tmp = a;
        rungekutta4bg(tmp, fourpiG,
          #ifdef HAVE_HICLASS_BG
            H_spline, acc,
          #else
            cosmo,
          #endif
          0.5 * dtau);
        rungekutta4bg(tmp, fourpiG,
          #ifdef HAVE_HICLASS_BG
            H_spline, acc,
          #else
            cosmo,
          #endif
          0.5 * dtau);

    		if (pkcount < sim.num_pk && 1. / tmp < sim.z_pk[pkcount] + 1.)
    		{
    		 writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
    					class_background, class_perturbs, ic,
      #ifdef HAVE_HICLASS_BG
         H_spline, acc, gsl_spline_eval(rho_smg_spline, a, acc), gsl_spline_eval(rho_crit_spline, 1., acc),
      #endif
#endif
    					&pcls_cdm, &pcls_b, pcls_ncdm, &phi,&pi_k, &zeta_half, &chi, &Bi,&T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT ,&scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij
    #ifdef CHECK_B
    					, &Bi_check, &BiFT_check, &plan_Bi_check
    #endif
    #ifdef VELOCITY
    				, &vi, &viFT, &plan_vi
    #endif
		    );
    #ifdef NONLINEAR_TEST
    writeSpectra_PoissonTerms(sim,  cosmo,  fourpiG,  a, pkcount,
    #ifdef HAVE_HICLASS_BG
    H_spline, acc,
    #endif
    &short_wave, &short_wave_scalarFT , &short_wave_plan);
    writeSpectra_phi_prime(sim, cosmo, fourpiG, a, pkcount,
    #ifdef HAVE_HICLASS_BG
    H_spline, acc,
    #endif
    &phi_prime, &phi_prime_scalarFT, &phi_prime_plan);
    #endif
    		}
  #endif // EXACT_OUTPUT_REDSHIFTS


#ifdef BENCHMARK
		spectra_output_time += MPI_Wtime() - ref_time;
#endif

		if (pkcount >= sim.num_pk && snapcount >= sim.num_snapshot)
		{
			for (i = 0; i < sim.num_lightcone; i++)
			{
				if (sim.lightcone[i].z + 1. < 1. / a)
					i = sim.num_lightcone + 1;
			}
			if (i == sim.num_lightcone) break; // simulation complete
		}

		// compute number of step subdivisions for ncdm particle updates
		for (i = 0; i < cosmo.num_ncdm; i++)
		{
			if (dtau * maxvel[i+1+sim.baryon_flag] > dx * sim.movelimit)
				numsteps_ncdm[i] = (int) ceil(dtau * maxvel[i+1+sim.baryon_flag] / dx / sim.movelimit);
			else numsteps_ncdm[i] = 1;
		}

		if (cycle % CYCLE_INFO_INTERVAL == 0)
		{
			COUT << " cycle " << cycle << ", time integration information: max |v| = " << maxvel[0] << " (cdm Courant factor = " << maxvel[0] * dtau / dx;
			if (sim.baryon_flag)
			{
				COUT << "), baryon max |v| = " << maxvel[1] << " (Courant factor = " << maxvel[1] * dtau / dx;
			}

      COUT << "), time step / Hubble time = " << Hconf(a, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc
      #else
        cosmo
      #endif
      ) * dtau;

			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (i == 0)
				{
					COUT << endl << " time step subdivision for ncdm species: ";
				}
				COUT << numsteps_ncdm[i] << " (max |v| = " << maxvel[i+1+sim.baryon_flag] << ")";
				if (i < cosmo.num_ncdm-1)
				{
					COUT << ", ";
				}
			}

			COUT << endl;
		}

		//Kessence
#ifdef BENCHMARK
		ref_time = MPI_Wtime();
#endif

        // We just need to update halo when we want to calculate spatial derivative or use some neibours at the same time! So here wo do not nee to update halo for phi_prime!

#ifdef HAVE_HICLASS_BG // If we have BG vlaues from hicalss/CLASS!
//**********************
//Kessence - LeapFrog:START
//**********************
// hideous bug if solving for cycle = 0
if (dtau_old>0.0){
	double a_kess = a; // Scale factor used for k_essence update.
	bool evolve_zeta_integer; // Used for updating zeta_integer.

	// zeta_half is allowed to be evolved backwards in time in the first cycle, i.e., cycle == 1, to get the correct half step. 
	// see thesis for details

	// Evolving zeta_half back one half k_essence step.
	if (cycle==1){
		avg_pi = average_func(  pi_k, Hconf(a_kess, fourpiG,
	  				#ifdef HAVE_HICLASS_BG
	  				H_spline,acc
	  				#else
	  				cosmo
	  				#endif
	  				), numpts3d );

		avg_zeta = average_func(  zeta_half,1., numpts3d );

				//COUT << "average pi =   "<< avg_pi <<endl;
		max_abs_pi = max_abs_func(pi_k, Hconf(a_kess, fourpiG,
	  				#ifdef HAVE_HICLASS_BG
	  				H_spline,acc
	  				#else
	  				cosmo
	  				#endif
	  				));
		max_abs_zeta = max_abs_func(zeta_half,1.);
		//double avg_phi = average_func(  phi,1., numpts3d );

		div_variables <<  1./(a_kess) -1. <<"          "<< avg_pi <<"        " << max_abs_pi<<"        "<< avg_zeta<<"       "<< max_abs_zeta <<endl;


		update_zeta_eq(-1./(sim.nKe_numsteps*2.) * dtau, dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
    	zeta_half.updateHalo();
		//avg_zeta = average_func(  zeta_half,1., numpts3d );
		//COUT << "avg_phi = "<<avg_phi<< endl;
		//COUT << "avg_zeta = "<<avg_zeta<< endl;



	}
	else{
		if (sim.blowup_criteria_met==false){ // Constant N_kess.  If new time step is smaller than old timestep, zeta_half must be evolved forwards
		                                     //    to the correct half step. Blowup has not happened here.
			if (dtau < dtau_old){
				update_zeta_eq(1./(sim.nKe_numsteps*2.) * (-dtau + dtau_old), dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
    			zeta_half.updateHalo();
			}
		}
		else{
			if (sim.kess_inner_loop_check==false){ // Blowup happened in the last iteration of the previous k_ess update. Both time step and N_kess may vary here.
				if (dtau_old/sim.nKe_numsteps > dtau/sim.new_nKe_numsteps){	
					update_zeta_eq( dtau_old/(2.*sim.nKe_numsteps) - dtau/(2.*sim.new_nKe_numsteps)  , dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
    				zeta_half.updateHalo();
				}
			}
			else{ 
				if (dtau < dtau_old){ // Constant N_kess. Blowup happened inside the k_ess loop of the previous cycle. We need only account for the change in cycle time step here.
					update_zeta_eq(1./(sim.new_nKe_numsteps*2.) * (-dtau + dtau_old), dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
    				zeta_half.updateHalo();
				}
			}
		} 
	}

	// k-essence loop (the (i) loop)
	for (i=0;i<sim.nKe_numsteps;i++){

		// This section activates if blowup has happened.
		#ifdef NONLINEAR_TEST
		if (sim.blowup_criteria_met){
			// Aborting if we have all the snapshots...

			if (sim.kess_inner_loop_check == false) sim.kess_inner_loop_check_func(true);
			
			// Finding the correct number of iterations to conserve the global (cycle) timestep.
			int num_j_iterations = sim.new_nKe_numsteps*(1. - (i+0.)/sim.nKe_numsteps); 
			
			// Evolving zeta_half to correct time step. Need only account for the change in N_kess. 
			if ((sim.new_nKe_numsteps > sim.nKe_numsteps) && (i>0)){ // For i==0, the pre evolution of zeta_half is done before the (i) loop.
				update_zeta_eq(dtau/2.*(1./sim.nKe_numsteps - 1./sim.new_nKe_numsteps), dx, a_kess, phi_prime,phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
				zeta_half.updateHalo();	
			}
			// k-essece blowup loop (the (j) loop)
			for (int j=0;j < num_j_iterations ;j++){
				// evolve and write fields
				//COUT << "cycle = " << cycle <<  "   i = " << i<< "   j = " << j << endl;

//				if (!(sim.snapcount_b <= sim.num_snapshot_kess)){
//					COUT << "Aborting as we have all the kess_snapshots..." << endl;
//	  		    	if(parallel.isRoot()) parallel.abortForce();
//	  			}
				

		    	update_zeta_eq(dtau/ sim.new_nKe_numsteps, dx, a_kess, phi_prime,phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
		    	zeta_half.updateHalo();
		    	rungekutta4bg(a_kess, fourpiG, H_spline, acc, dtau  / sim.new_nKe_numsteps / 2.0);
		    	//********************************************************************************
		    	//we update pi to have it at n+1 (at first loop from the value at (0) and the value of zeta_integer at 1/2 and H(n+1/2) we update pi at (1))
		    	//In the pi update we also update zeta_int because we need the values of a_kess and H_kess at step n+1/2
		    	//By the below update we get pi(n+1) and zeta(n+1)
		    	//********************************************************************************
		    	//if (i!=0)
				update_pi_eq(dtau/ sim.new_nKe_numsteps,dtau_old,dtau,phi,chi, phi_old, chi_old, pi_k, zeta_half, Hconf(a_kess, fourpiG, H_spline, acc));//,kessence_iteration_loop_in_loop + j); // H_old is updated here in the function
				//else
				//update_pi_eq(dtau/ sim.nKe_numsteps,psi_prime, phi_old, chi_old, pi_k, zeta_half, Hconf(a_kess, fourpiG, H_spline, acc),sim.nKe_numsteps/old_nKe_numsteps*(i+1.) + j);

				pi_k.updateHalo();

		    	//********************************************************************************
		    	// Now we have pi(n+1) and a_kess(n+1/2) so we update background by halfstep to have a_kess(n+1)
		    	//********************************************************************************
		    	rungekutta4bg(a_kess, fourpiG,
		    	  #ifdef HAVE_HICLASS_BG
		    	    H_spline, acc,
		    	  #else
		    	    cosmo,
		    	  #endif
		    	  dtau  / sim.new_nKe_numsteps / 2.0 );


		    	avg_pi = average_func(  pi_k, Hconf(a_kess, fourpiG,
	  				#ifdef HAVE_HICLASS_BG
	  				H_spline,acc
	  				#else
	  				cosmo
	  				#endif
	  				), numpts3d );

				avg_zeta = average_func(  zeta_half,1., numpts3d );

				//COUT << "average pi =   "<< avg_pi <<endl;
				max_abs_pi = max_abs_func(pi_k, Hconf(a_kess, fourpiG,
	  				#ifdef HAVE_HICLASS_BG
	  				H_spline,acc
	  				#else
	  				cosmo
	  				#endif
	  				));
				max_abs_zeta = max_abs_func(zeta_half,1.);
				div_variables <<  1./(a_kess) -1. <<"          "<< avg_pi <<"        " << max_abs_pi<<"        "<< avg_zeta<<"       "<< max_abs_zeta << endl;


				if ( sim.snapcount_b <= sim.num_snapshot_kess ){
				// Updating zeta_integer field to same time as pi_k.
				update_zeta_eq(dtau/ sim.new_nKe_numsteps / 2., dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=true,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
				zeta_integer.updateHalo();
		      
		      //if(parallel.isRoot())  cout << "\033[1;32mThe blowup criteria are met, the requested snapshots being produced\033[0m\n";
			     //std::string output_path_string = sim.output_path;

		     	  //std::string output_path_string = sim.output_path;

		        // writeSpectra(sim, cosmo, fourpiG, a, snapcount_b,
		        //           &pcls_cdm, &pcls_b, pcls_ncdm, &phi,&pi_k, &zeta_half, &chi, &Bi,&T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT ,&scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij);
			  std::string output_path_string = sim.output_path;
			  str_filename =  output_path_string + "pi_k_" + to_string(sim.snapcount_b) + ".h5";
		          //str_filename2 = output_path_string + "zeta_" + to_string(snapcount_b) + ".h5";
		          //str_filename3 = output_path_string + "phi_" + to_string(snapcount_b) + ".h5";
		      pi_k.saveHDF5(str_filename);
			  #ifdef FLUID_VARIABLES
			   Hc = Hconf(a_kess, fourpiG,
			  #ifdef HAVE_HICLASS_BG
			  H_spline, acc
			  #else
			  cosmo
			  #endif
			  );
			  
        	  calculate_fluid_properties(div_v_upper_fluid,Sigma_upper_ij_fluid, delta_rho_fluid,delta_p_fluid,v_upper_i_fluid,pi_k,zeta_integer,phi,chi,gsl_spline_eval(rho_smg_spline, a_kess, acc)
        			,gsl_spline_eval(p_smg_spline, a_kess, acc),gsl_spline_eval(cs2_spline, a_kess, acc), Hc,dx,a_kess,gsl_spline_eval(rho_crit_spline, 1., acc), Gevolution_H0);
        	  
              str_filename =  output_path_string + "delta_rho_fluid_" + to_string(sim.snapcount_b) + ".h5";
        	  delta_rho_fluid.saveHDF5(str_filename);
        	  str_filename =  output_path_string + "delta_p_fluid_" + to_string(sim.snapcount_b) + ".h5";
        	  delta_p_fluid.saveHDF5(str_filename);
        	  str_filename =  output_path_string + "v_upper_i_fluid_" + to_string(sim.snapcount_b) + ".h5";
        	  v_upper_i_fluid.saveHDF5(str_filename);
			  ////str_filename =  output_path_string + "Sigma_upper_ij_fluid_" + to_string(sim.snapcount_b) + ".h5";
			  ////Sigma_upper_ij_fluid.saveHDF5(str_filename);
			  str_filename =  output_path_string + "div_v_upper_fluid_" + to_string(sim.snapcount_b) + ".h5";
        	  div_v_upper_fluid.saveHDF5(str_filename);
        		
        	  #endif
		          //zeta_half.saveHDF5(str_filename2);
		          //phi.saveHDF5(str_filename3);
		          // str_filename =  "./output/pi_k_" + to_string(snapcount_b-1) + ".h5";
		          // str_filename2 = "./output/zeta_" + to_string(snapcount_b-1) + ".h5";
		          // pi_k_old.saveHDF5(str_filename);
		          // zeta_half_old.saveHDF5(str_filename2);
		          

		        //****************************
		        //****PRINTING snapshots info
		        //****************************
		          // COUT << scientific << setprecision(8);
		          // if(parallel.isRoot())
		          // {
		          // out_snapshots<<"### 1- tau\t2- z \t3- a\t 4- zeta_avg\t 5- avg_pi\t 6- avg_phi\t 7- tau/boxsize\t 8- H_conf/H0 \t 9- snap_count"<<endl;

		      kess_snapshots<<setw(9) << tau + dtau/sim.nKe_numsteps*(i+0.) + dtau/sim.new_nKe_numsteps*(j+1.) <<"\t"<< setw(9) << 1./(a_kess) -1.0 <<"\t"<< setw(9) << a_kess <<"\t"<< setw(9) << avg_zeta <<"\t"<< setw(9) << avg_pi <<"\t"<< setw(9) << avg_phi <<"\t"<< setw(9) <<tau <<"\t"<< setw(9) <<Hconf(a_kess, fourpiG,
				#ifdef HAVE_HICLASS_BG
			        H_spline,acc
			    #else
			        cosmo
			    #endif 
				) / Hconf(1., fourpiG,
				#ifdef HAVE_HICLASS_BG
			      H_spline, acc
			    #else
		          cosmo
			    #endif 
				) << "\t"<< setw(9) <<sim.snapcount_b  <<endl;
			  sim.snapcount_b_add_one();
			}
				  
			if (std::isnan(avg_zeta)){
				rho_i_rho_crit_0.close();
				//convert_to_cosmic_velocity.close();
				Result_avg.close();
  				Result_real.close();
				Result_fourier.close();
  				Result_max.close();
  				Redshifts.close();
  				kess_snapshots.close();
  				div_variables.close();
				potentials.close();
				COUT << "avg_zeta is nan" << endl;

				if(parallel.isRoot()) parallel.abortForce();
			}


				}// j loop
			break; // breaking the i loop.
		}// if test
		#endif // NONLINEAR_TEST

    update_zeta_eq(dtau/ sim.nKe_numsteps, dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=false,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
    zeta_half.updateHalo();
    //rungekutta4bg(a_kess, fourpiG, H_spline, acc, dtau  / sim.nKe_numsteps / 2.0);
	rungekutta4bg(a_kess, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc,
      #else
        cosmo,
      #endif
      dtau  / sim.nKe_numsteps / 2.0 );
    //********************************************************************************
    //we update pi to have it at n+1 (at first loop from the value at (0) and the value of zeta_integer at 1/2 and H(n+1/2) we update pi at (1))
    //In the pi update we also update zeta_int because we need the values of a_kess and H_kess at step n+1/2
    //By the below update we get pi(n+1) and zeta(n+1)
    //********************************************************************************
    update_pi_eq(dtau/ sim.nKe_numsteps, dtau_old,dtau,phi,chi, phi_old, chi_old, pi_k, zeta_half, Hconf(a_kess, fourpiG, H_spline, acc));//,i); // H_old is updated here in the function
	pi_k.updateHalo();

    //********************************************************************************
    // Now we have pi(n+1) and a_kess(n+1/2) so we update background by halfstep to have a_kess(n+1)
    //********************************************************************************
    rungekutta4bg(a_kess, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc,
      #else
        cosmo,
      #endif
      dtau  / sim.nKe_numsteps / 2.0 );

      // NL TESTS
    #ifdef NONLINEAR_TEST
      //   //Make snapshots and power around blowup TIME
	  //std::string output_path_string = sim.output_path;
	  //std::cout << "cycle = " << cycle << std::endl;
	  //std::cout << "before test" << std::endl;
	//avg_zeta =average_func(  zeta_half,1., numpts3d ) ;
	      //avg_zeta_old =average(  zeta_half_old,1., numpts3d ) ;
    //avg_pi =average(  pi_k,1., numpts3d ) ;
	  
      //avg_phi =average(  phi , 1., numpts3d ) ;
      // avg_pi_old =average(  pi_k_old, 1., numpts3d ) ;

    avg_pi = average_func(  pi_k, Hconf(a_kess, fourpiG,
	  #ifdef HAVE_HICLASS_BG
	  H_spline,acc
	  #else
	  cosmo
	  #endif
	  ), numpts3d );

	avg_zeta = average_func(  zeta_half,1., numpts3d );

	max_abs_pi = max_abs_func(pi_k, Hconf(a_kess, fourpiG,
	  #ifdef HAVE_HICLASS_BG
	  H_spline,acc
	  #else
	  cosmo
	  #endif
	  ));
	max_abs_zeta = max_abs_func(zeta_half,1.);
	div_variables <<  1./(a_kess) -1. <<"          "<< avg_pi <<"        " << max_abs_pi<<"        "<< avg_zeta<<"       "<< max_abs_zeta << endl;
	

    

	  // This if test checks if the dark energy field is blowing up.
	  //if ((max_abs_zeta > 1000.*avg_zeta && avg_zeta > 1e-7) || (sim.kess_inner_loop_check) || (1./(a_kess) -1.0 <= sim.known_blowup_time) || (abs(avg_pi) > 1.)){
	if ((1./(a_kess) -1.0 <= sim.known_blowup_time) || (max_abs_zeta > 0.001) ){

	  // Blowup has happened
	  sim.blowup_func(true);  	  
	  
      // Aborting if we have all the snapshots...
//	  if (!(sim.snapcount_b <= sim.num_snapshot_kess)){
//		COUT << "Aborting as we have all the kess_snapshots..." << endl;
//	    if(parallel.isRoot()) parallel.abortForce();
//	  }
      COUT << "\033[1;32mThe blowup criteria are met, the requested snapshots being produced\033[0m\n";
	   div_variables << "### The blowup criteria are met"<<endl;
	  
	  if(sim.snapcount_b <= sim.num_snapshot_kess){
		// updating zeta_integer field to n+1 (same as pi_k)
		update_zeta_eq(dtau/ sim.nKe_numsteps / 2., dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer,evolve_zeta_integer=true,  gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
		zeta_integer.updateHalo();
	    std::string output_path_string = sim.output_path;

     	  //std::string output_path_string = sim.output_path;

        // writeSpectra(sim, cosmo, fourpiG, a, snapcount_b,
        //           &pcls_cdm, &pcls_b, pcls_ncdm, &phi,&pi_k, &zeta_half, &chi, &Bi,&T00_Kess, &T0i_Kess, &Tij_Kess, &source, &Sij, &scalarFT ,&scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_KessFT, &T0i_KessFT, &Tij_KessFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_Kess, &plan_T0i_Kess, &plan_Tij_Kess, &plan_source, &plan_Sij);
          //str_filename2 = output_path_string + "zeta_" + to_string(snapcount_b) + ".h5";
          //str_filename3 = output_path_string + "phi_" + to_string(snapcount_b) + ".h5";
		
		
		str_filename =  output_path_string + "pi_k_" + to_string(sim.snapcount_b) + ".h5";
		pi_k.saveHDF5(str_filename);

		#ifdef FLUID_VARIABLES
			Hc = Hconf(a_kess, fourpiG,
			  #ifdef HAVE_HICLASS_BG
			    H_spline, acc
			  #else
			    cosmo
			  #endif
			    );
			
			calculate_fluid_properties(div_v_upper_fluid,Sigma_upper_ij_fluid, delta_rho_fluid,delta_p_fluid,v_upper_i_fluid,pi_k,zeta_integer,phi,chi,gsl_spline_eval(rho_smg_spline, a_kess, acc)
				,gsl_spline_eval(p_smg_spline, a_kess, acc),gsl_spline_eval(cs2_spline, a_kess, acc), Hc,dx,a_kess,gsl_spline_eval(rho_crit_spline, 1., acc), Gevolution_H0);

			str_filename =  output_path_string + "delta_rho_fluid_" + to_string(sim.snapcount_b) + ".h5";
			delta_rho_fluid.saveHDF5(str_filename);
			str_filename =  output_path_string + "delta_p_fluid_" + to_string(sim.snapcount_b) + ".h5";
			delta_p_fluid.saveHDF5(str_filename);
			str_filename =  output_path_string + "v_upper_i_fluid_" + to_string(sim.snapcount_b) + ".h5";
			v_upper_i_fluid.saveHDF5(str_filename);
			////str_filename =  output_path_string + "Sigma_upper_ij_fluid_" + to_string(sim.snapcount_b) + ".h5";
			////Sigma_upper_ij_fluid.saveHDF5(str_filename);
			str_filename =  output_path_string + "div_v_upper_fluid_" + to_string(sim.snapcount_b) + ".h5";
			div_v_upper_fluid.saveHDF5(str_filename);
			
			#endif


        //****************************
        //****PRINTING snapshots info
        //****************************
          // COUT << scientific << setprecision(8);
          // if(parallel.isRoot())
          // {
          // out_snapshots<<"### 1- tau\t2- z \t3- a\t 4- zeta_avg\t 5- avg_pi\t 6- avg_phi\t 7- tau/boxsize\t 8- H_conf/H0 \t 9- snap_count"<<endl;

        kess_snapshots<<setw(9) << tau + dtau/sim.nKe_numsteps*(i+1.) <<"\t"<< setw(9) << 1./(a_kess) -1.0 <<"\t"<< setw(9) << a_kess <<"\t"<< setw(9) << avg_zeta <<"\t"<< setw(9) << avg_pi <<"\t"<< setw(9) << avg_phi <<"\t"<< setw(9) <<tau <<"\t"<< setw(9) <<Hconf(a_kess, fourpiG,
		#ifdef HAVE_HICLASS_BG
	        H_spline,acc
	    #else
	        cosmo
	    #endif 
		) / Hconf(1., fourpiG,
		#ifdef HAVE_HICLASS_BG
	      H_spline, acc
	    #else
          cosmo
	    #endif 
		)  << "\t"<< setw(9) << sim.snapcount_b  << endl;
        sim.snapcount_b_add_one();
		}
	  }// if test i loop
		#endif
	}// i loop

// updating the zeta_integer field to n+1. The integer field is always evolved from the zeta_half field...
if (sim.kess_inner_loop_check){
	update_zeta_eq(dtau/ sim.new_nKe_numsteps / 2., dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer ,evolve_zeta_integer=true, gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
	zeta_integer.updateHalo();
}
else{
	update_zeta_eq(dtau/ sim.nKe_numsteps / 2., dx, a_kess,phi_prime, phi, chi, pi_k, zeta_half,zeta_integer ,evolve_zeta_integer=true, gsl_spline_eval(cs2_spline, a_kess, acc), gsl_spline_eval(cs2_prime_spline, a_kess, acc)/gsl_spline_eval(cs2_spline, a_kess, acc)/(a_kess* gsl_spline_eval(H_spline, a_kess, acc)),  gsl_spline_eval(p_smg_prime_spline, a_kess, acc)/gsl_spline_eval(rho_smg_prime_spline, a_kess, acc), Hconf(a_kess, fourpiG, H_spline, acc), Hconf_prime(a_kess, fourpiG, H_spline, acc), sim.NL_kessence);
	zeta_integer.updateHalo();
}

#ifdef BENCHMARK
    kessence_update_time += MPI_Wtime() - ref_time;
    ref_time = MPI_Wtime();
#endif
	
}

#else // If not HAVE_HICLASS_BG We use  kevolution with w, c_s^2 constants.
//**********************
//Kessence - LeapFrog:START
//**********************
  double a_kess=a;
  //First we update zeta_half to have it at -1/2 just in the first loop
  if(cycle==0)
  {
    for (i=0;i<sim.nKe_numsteps;i++)
    {
      //computing zeta_half(-1/2) and zeta_int(-1) but we do not work with zeta(-1)
      update_zeta(-dtau/ (2. * sim.nKe_numsteps) , dx, a_kess, phi, phi_old, chi, chi_old, pi_k, zeta_half, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hconf(a_kess, fourpiG, cosmo), Hconf_prime(a_kess, fourpiG, cosmo), sim.NL_kessence);
      // zeta_integer.updateHalo();
      zeta_half.updateHalo();
    }
  }

 //Then fwe start the main loop zeta is updated to get zeta(n+1/2) from pi(n) and zeta(n-1/2)
	for (i=0;i<sim.nKe_numsteps;i++)
	{
    //********************************************************************************
    //Updating zeta_integer to get zeta_integer(n+1/2) and zeta_integer(n+1), in the first loop is getting zeta_integer(1/2) and zeta_integer(1)
    // In sum: zeta_integer(n+1/2) = zeta_integer(n-1/2)+ zeta_integer'(n)dtau which needs background to be at n with then
    //Note that here for zeta_integer'(n) we need background to be at n and no need to update it.
    //\zeta_integer(n+1/2) = \zeta_integer(n-1/2) + \zeta_integer'(n)  dtau
    //We also update zeta_int from n to n+1
    //********************************************************************************
    update_zeta(dtau/ sim.nKe_numsteps, dx, a_kess, phi, phi_old, chi, chi_old, pi_k, zeta_half, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hconf(a_kess, fourpiG, cosmo), Hconf_prime(a_kess, fourpiG, cosmo), sim.NL_kessence);
    // zeta_integer.updateHalo();
    zeta_half.updateHalo();
    //********************************************************************************
    //Since we have pi(n+1)=pi(n) + pi'(n+1/2), and in pi'(n+1/2) we have H(n+1/2) we update the background before updating the pi to have H(n+1/2), Moreover zeta(n+1) = zeta(n+1/2) + zeta'(n+1/2), so we put zeta_int updating in the pi updating!
    //********************************************************************************
    rungekutta4bg(a_kess, fourpiG, cosmo, dtau  / sim.nKe_numsteps / 2.0);
    //********************************************************************************
    //we update pi to have it at n+1 (at first loop from the value at (0) and the value of zeta_integer at 1/2 and H(n+1/2) we update pi at (1))
    //In the pi update we also update zeta_int because we need the values of a_kess and H_kess at step n+1/2
    //By the below update we get pi(n+1) and zeta(n+1)
    //********************************************************************************
    update_pi_k(dtau/ sim.nKe_numsteps, dx, a_kess, phi, phi_old, chi, chi_old, pi_k, zeta_half, cosmo.Omega_kessence, cosmo.w_kessence, cosmo.cs2_kessence, Hconf(a_kess, fourpiG, cosmo), Hconf_prime(a_kess, fourpiG, cosmo), sim.NL_kessence); // H_old is updated here in the function
		pi_k.updateHalo();

    //********************************************************************************
    // Now we have pi(n+1) and a_kess(n+1/2) so we update background by halfstep to have a_kess(n+1)
    //********************************************************************************
    rungekutta4bg(a_kess, fourpiG, cosmo, dtau  / sim.nKe_numsteps / 2.0 );
      #ifdef NONLINEAR_TEST
      avg_zeta =average(  zeta_half,1., numpts3d ) ;
      avg_zeta_old =average(  zeta_half_old,1., numpts3d ) ;
      avg_pi =average(  pi_k,1., numpts3d ) ;
      avg_phi =average(  phi , 1., numpts3d ) ;

      if ( avg_zeta > 1.e-7 && abs(avg_zeta/avg_zeta_old)>1.02 && sim.snapcount_b< sim.num_snapshot_kess )
      {
      if(parallel.isRoot())  cout << "\033[1;32mThe blowup criteria are met, the requested snapshots being produced\033[0m\n";
          str_filename =  "../output/pi_k_" + to_string(sim.snapcount_b) + ".h5";
          str_filename2 = "../output/zeta_" + to_string(sim.snapcount_b) + ".h5";
          str_filename3 = "../output/phi_" + to_string(sim.snapcount_b) + ".h5";
          pi_k.saveHDF5(str_filename);
          zeta_half.saveHDF5(str_filename2);
          phi.saveHDF5(str_filename3);
          sim.snapcount_b_add_one();
          kess_snapshots<<setw(9) << tau + dtau/sim.nKe_numsteps <<"\t"<< setw(9) << 1./(a_kess) -1.0 <<"\t"<< setw(9) << a_kess <<"\t"<< setw(9) << avg_zeta <<"\t"<< setw(9) << avg_pi <<"\t"<< setw(9) << avg_phi <<"\t"<< setw(9) <<tau <<"\t"<< setw(9) <<Hconf(a_kess, fourpiG, cosmo) / Hconf(1., fourpiG, cosmo)<<"\t"<< setw(9) <<sim.snapcount_b  <<endl;
        }
    #endif
    }
#ifdef BENCHMARK
    kessence_update_time += MPI_Wtime() - ref_time;
    ref_time = MPI_Wtime();
#endif
#endif

//**********************
//Kessence - LeapFrog: End
//**********************

#ifdef BENCHMARK
		ref2_time = MPI_Wtime();
#endif
		for (i = 0; i < cosmo.num_ncdm; i++) // non-cold DM particle update
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;

			tmp = a;

			for (j = 0; j < numsteps_ncdm[i]; j++)
			{
				f_params[0] = tmp;
				f_params[1] = tmp * tmp * sim.numpts;
				if (sim.gr_flag > 0)
					maxvel[i+1+sim.baryon_flag] = pcls_ncdm[i].updateVel(update_q, (dtau + dtau_old) / 2. / numsteps_ncdm[i], update_ncdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
				else
					maxvel[i+1+sim.baryon_flag] = pcls_ncdm[i].updateVel(update_q_Newton, (dtau + dtau_old) / 2. / numsteps_ncdm[i], update_ncdm_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);

#ifdef BENCHMARK
				update_q_count++;
				update_q_time += MPI_Wtime() - ref2_time;
				ref2_time = MPI_Wtime();
#endif

				rungekutta4bg(tmp, fourpiG,
        #ifdef HAVE_HICLASS_BG
          H_spline, acc,
        #else
          cosmo,
        #endif
        0.5 * dtau / numsteps_ncdm[i]);
				f_params[0] = tmp;
				f_params[1] = tmp * tmp * sim.numpts;

				if (sim.gr_flag > 0)
					pcls_ncdm[i].moveParticles(update_pos, dtau / numsteps_ncdm[i], update_ncdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
				else
					pcls_ncdm[i].moveParticles(update_pos_Newton, dtau / numsteps_ncdm[i], NULL, 0, f_params);
#ifdef BENCHMARK
				moveParts_count++;
				moveParts_time += MPI_Wtime() - ref2_time;
				ref2_time = MPI_Wtime();
#endif
				rungekutta4bg(tmp, fourpiG,
          #ifdef HAVE_HICLASS_BG
            H_spline, acc,
          #else
            cosmo,
          #endif
          0.5 * dtau / numsteps_ncdm[i]);
			}
		}

		// cdm and baryon particle update
		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			maxvel[0] = pcls_cdm.updateVel(update_q, (dtau + dtau_old) / 2., update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q, (dtau + dtau_old) / 2., update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
		}
		else
		{
			maxvel[0] = pcls_cdm.updateVel(update_q_Newton, (dtau + dtau_old) / 2., update_cdm_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q_Newton, (dtau + dtau_old) / 2., update_b_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
		}

#ifdef BENCHMARK
		update_q_count++;
		update_q_time += MPI_Wtime() - ref2_time;
		ref2_time = MPI_Wtime();
#endif
	
	#ifdef FLUID_VARIABLES
	a_old_for_kess_velocity = a;
	#endif
      rungekutta4bg(a, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc,
      #else
        cosmo,
      #endif
      0.5 * dtau);  // evolve background by half a time step

		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			pcls_cdm.moveParticles(update_pos, dtau, update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos, dtau, update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
		}
		else
		{
			pcls_cdm.moveParticles(update_pos_Newton, dtau, NULL, 0, f_params);
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos_Newton, dtau, NULL, 0, f_params);
		}

#ifdef BENCHMARK
		moveParts_count++;
		moveParts_time += MPI_Wtime() - ref2_time;
#endif

    rungekutta4bg(a, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc,
      #else
        cosmo,
      #endif
      0.5 * dtau);  // evolve background by half a time step

		parallel.max<double>(maxvel, numspecies);

		if (sim.gr_flag > 0)
		{
			for (i = 0; i < numspecies; i++)
				maxvel[i] /= sqrt(maxvel[i] * maxvel[i] + 1.0);
		}
		// done particle update

		tau += dtau;

		if (sim.wallclocklimit > 0.)   // check for wallclock time limit
		{
			tmp = MPI_Wtime() - start_time;
			parallel.max(tmp);
			if (tmp > sim.wallclocklimit)   // hibernate
			{
				COUT << COLORTEXT_YELLOW << " reaching hibernation wallclock limit, hibernating..." << COLORTEXT_RESET << endl;
				COUT << COLORTEXT_CYAN << " writing hibernation point" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
				if (sim.vector_flag == VECTOR_PARABOLIC && sim.gr_flag == 0)
					plan_Bi.execute(FFT_BACKWARD);
#ifdef CHECK_B
				if (sim.vector_flag == VECTOR_ELLIPTIC)
				{
					plan_Bi_check.execute(FFT_BACKWARD);
					hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, pi_k, zeta_half, chi, Bi_check, a, tau, dtau, cycle);
				}
				else
#endif
				hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, pi_k, zeta_half, chi, Bi, a, tau, dtau, cycle);
				break;
			}
		}

		if (restartcount < sim.num_restart && 1. / a < sim.z_restart[restartcount] + 1.)
		{
			COUT << COLORTEXT_CYAN << " writing hibernation point" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
			if (sim.vector_flag == VECTOR_PARABOLIC && sim.gr_flag == 0)
				plan_Bi.execute(FFT_BACKWARD);
#ifdef CHECK_B
			if (sim.vector_flag == VECTOR_ELLIPTIC)
			{
				plan_Bi_check.execute(FFT_BACKWARD);
				hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, pi_k, zeta_half, chi, Bi, a, tau, dtau, cycle, restartcount);
			}
			else
#endif
			hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, pi_k, zeta_half, chi, Bi, a, tau, dtau, cycle, restartcount);
			restartcount++;
		}

		dtau_old = dtau;

    if (sim.Cf * dx < sim.steplimit / Hconf(a, fourpiG,
    #ifdef HAVE_HICLASS_BG
      H_spline, acc
    #else
      cosmo
    #endif
    ))
			dtau = sim.Cf * dx;
		else
      dtau = sim.steplimit / Hconf(a, fourpiG,
      #ifdef HAVE_HICLASS_BG
        H_spline, acc
      #else
        cosmo
      #endif
      );
	  
	  if (dtau > dtau_old) dtau = dtau_old;
	  //COUT << cycle<<"   "<<dtau_old<<"   "<<dtau << endl;

		cycle++;

#ifdef BENCHMARK
		cycle_time += MPI_Wtime()-cycle_start_time;
#endif
	}
	// closing the files
#ifdef NONLINEAR_TEST
    rho_i_rho_crit_0.close();
	//convert_to_cosmic_velocity.close();
	Result_avg.close();
  	Result_real.close();
	Result_fourier.close();
  	Result_max.close();
  	Redshifts.close();
  	kess_snapshots.close();
  	div_variables.close();
	potentials.close();
	//Omega.close();
#endif

	COUT << COLORTEXT_GREEN << " simulation complete." << COLORTEXT_RESET << endl;

#ifdef BENCHMARK
		ref_time = MPI_Wtime();
#endif

#if defined(HAVE_CLASS) || defined(HAVE_HICLASS)
	if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
		freeCLASSstructures(class_background, class_thermo, class_perturbs);
#endif

#ifdef BENCHMARK
	lightcone_output_time += MPI_Wtime() - ref_time;
	run_time = MPI_Wtime() - start_time;

	parallel.sum(run_time);
	parallel.sum(cycle_time);
	parallel.sum(projection_time);
	parallel.sum(snapshot_output_time);
	parallel.sum(spectra_output_time);
	parallel.sum(lightcone_output_time);
	parallel.sum(gravity_solver_time);
  parallel.sum(kessence_update_time);
	parallel.sum(fft_time);
	parallel.sum(update_q_time);
	parallel.sum(moveParts_time);

	COUT << endl << "BENCHMARK" << endl;
	COUT << "total execution time  : "<<hourMinSec(run_time) << endl;
	COUT << "total number of cycles: "<< cycle << endl;
	COUT << "time consumption breakdown:" << endl;
	COUT << "initialization   : "  << hourMinSec(initialization_time) << " ; " << 100. * initialization_time/run_time <<"%."<<endl;
	COUT << "main loop        : "  << hourMinSec(cycle_time) << " ; " << 100. * cycle_time/run_time <<"%."<<endl;

	COUT << "----------- main loop: components -----------"<<endl;

	COUT << "projections                : "<< hourMinSec(projection_time) << " ; " << 100. * projection_time/cycle_time <<"%."<<endl;
  //Kessence update
  COUT << "Kessence_update                : "<< hourMinSec(kessence_update_time) << " ; " << 100. * kessence_update_time/cycle_time <<"%."<<endl;
	COUT << "snapshot outputs           : "<< hourMinSec(snapshot_output_time) << " ; " << 100. * snapshot_output_time/cycle_time <<"%."<<endl;
	COUT << "lightcone outputs          : "<< hourMinSec(lightcone_output_time) << " ; " << 100. * lightcone_output_time/cycle_time <<"%."<<endl;
	COUT << "power spectra outputs      : "<< hourMinSec(spectra_output_time) << " ; " << 100. * spectra_output_time/cycle_time <<"%."<<endl;
	COUT << "update momenta (count: "<<update_q_count <<"): "<< hourMinSec(update_q_time) << " ; " << 100. * update_q_time/cycle_time <<"%."<<endl;
	COUT << "move particles (count: "<< moveParts_count <<"): "<< hourMinSec(moveParts_time) << " ; " << 100. * moveParts_time/cycle_time <<"%."<<endl;
	COUT << "gravity solver             : "<< hourMinSec(gravity_solver_time) << " ; " << 100. * gravity_solver_time/cycle_time <<"%."<<endl;
	COUT << "-- thereof Fast Fourier Transforms (count: " << fft_count <<"): "<< hourMinSec(fft_time) << " ; " << 100. * fft_time/gravity_solver_time <<"%."<<endl;
#endif

#ifdef EXTERNAL_IO
		ioserver.stop();
	}
#endif
        //std::cout << Hconf() << std::endl;
	return 0;
}