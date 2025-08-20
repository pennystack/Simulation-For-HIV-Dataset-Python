#include <iostream>
#include <math.h> 
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

namespace py = pybind11;

// Helper function to get a vector from a py::dict
template<typename T>
std::vector<T> get_vector_from_dict(py::dict d, const char* key) {
    if (!d.contains(key)) {
        throw std::runtime_error("Dictionary is missing key: " + std::string(key));
    }
    return d[key].cast<std::vector<T>>();
}


double cpp_p(int i, int j, const std::vector<double>& aij, const std::vector<double>& bij, double t, int nstates) {
  double out = t * aij[j * nstates + i] + bij[j * nstates + i];
  double p_out = std::exp(out) / (1 + std::exp(out));
  
  if (p_out < 1e-100) {
    p_out = 1e-10;
  }
  
  //std::cout << "cpp_p: p_out is " << p_out << std::endl;
  return p_out;
}


double cpp_f(int i, int j, const std::vector<double>& vij, const std::vector<double>& sij, double x, int nstates) {
  double out = (vij[j * nstates + i] / sij[j * nstates + i]) * std::pow(x / sij[j * nstates + i], vij[j * nstates + i] - 1) * std::exp(-std::pow(x / sij[j * nstates + i], vij[j * nstates + i]));
  if (out < 1e-100) out = 1e-10;
  //std::cout << "cpp_f: out is " << out << std::endl;
  return out;
}


double cpp_F(int i, int j, const std::vector<double>& vij, const std::vector<double>& sij, double x, int nstates) {
  double out = 1 - std::exp(-std::pow(x / sij[j * nstates + i], vij[j * nstates + i]));
  if (out < 1e-100) out = 1e-10;
  if (out > 1) out = 1;
  //std::cout << "cpp_F: out is " << out << std::endl;
  return out;
}


double cpp_H(int i, double t, double x, const std::vector<double>& aij, const std::vector<double>& bij, const std::vector<double>& vij, const std::vector<double>& sij, int nstates) {
  
  double out = 0.0;
      for (int j_loop = 0; j_loop < nstates; j_loop++) {
        if (j_loop != i) {
            out += cpp_p(i, j_loop, aij, bij, t, nstates) * cpp_F(i, j_loop, vij, sij, x, nstates);
        }
    }
  
  if (out < 1e-100) out = 1e-10;
  if (out > 1) out = 1;
  //std::cout << "cpp_H: out is " << out << std::endl;
  return out;
}


double cpp_S(int i, double t, double x, const std::vector<double>& aij, const std::vector<double>& bij, const std::vector<double>& vij, const std::vector<double>& sij, int nstates) {
  
  double out = 1 - cpp_H(i, t, x, aij, bij, vij, sij, nstates);
  
  if (out < 1e-100) out = 1e-10;
  if (out > 1) out = 1;
  //std::cout << "cpp_S: out is " << out << std::endl;
  return out;
}


double cpp_loglikelihood(py::dict obstimes, const std::vector<double>& aij, const std::vector<double>& bij, const std::vector<double>& vij, const std::vector<double>& sij, int nstates) {
    std::vector<double> patients = get_vector_from_dict<double>(obstimes, "PATIENT");
    std::vector<double> state = get_vector_from_dict<double>(obstimes, "state");
    std::vector<double> obstime = get_vector_from_dict<double>(obstimes, "obstime");
    std::vector<double> deltaobstime = get_vector_from_dict<double>(obstimes, "deltaobstime");

    double out = 0.;
    size_t n = patients.size();
    
    for (size_t i = 0; i < n - 1; i++) {
      if (patients[i] == patients[i + 1]) {
        if (state[i] != state[i + 1]) {
            out += (log(cpp_p(state[i], state[i + 1], aij, bij, obstime[i], nstates)) + log(cpp_f(state[i], state[i + 1], vij, sij, deltaobstime[i], nstates))) ;
          } else {
              out += log(cpp_S(state[i], obstime[i], deltaobstime[i], aij, bij, vij, sij, nstates)) ;
          }
      }
    }
    
  //std::cout << "cpp_likelihood: out is " << out << std::endl;
  return out;
}


double cpp_c(int i, int j, const std::vector<double>& aij, const std::vector<double>& bij, const std::vector<double>& vij, const std::vector<double>& sij, double t, double x, int nstates) {
  double out = cpp_p(i, j, aij, bij, t, nstates) * cpp_f(i, j, vij, sij, x, nstates) ;
  if (out < 1e-100) out = 1e-100;
  //std::cout << "cpp_c: out is " << out << std::endl;

  return out;
}

PYBIND11_MODULE(hiv_smm, m) {
    m.def("cpp_p", &cpp_p, "pij");
    m.def("cpp_f", &cpp_f, "fij");
    m.def("cpp_F", &cpp_F, "Fij");
    m.def("cpp_H", &cpp_H, "Hij");
    m.def("cpp_S", &cpp_S, "Sij");
    m.def("cpp_loglikelihood", &cpp_loglikelihood, "loglikelihood_cpp");
    m.def("cpp_c", &cpp_c, "cij");
}