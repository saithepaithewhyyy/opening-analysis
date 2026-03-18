#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "classifier.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_classifier, m) {
    m.doc() = "Fast Bayesian chess opening classifier";

     py::class_<ScoredOpening>(m, "ScoredOpening")
          .def_readonly("eco",         &ScoredOpening::eco)
          .def_readonly("name",        &ScoredOpening::name)
          .def_readonly("likelihood",  &ScoredOpening::likelihood)
          .def_readonly("posterior",   &ScoredOpening::posterior)
          .def_readonly("path_length", &ScoredOpening::path_length)
          .def("__repr__", [](const ScoredOpening& s) {
               return "<" + s.eco + " " + s.name +
                    " posterior=" + std::to_string(s.posterior) +
                    " path="      + std::to_string(s.path_length) + ">";
          });

     py::class_<ClassifierEngine>(m, "ClassifierEngine")
          .def(py::init<>())
          .def("load_eco", &ClassifierEngine::load_eco,     py::arg("rows"))
          .def("load_priors", &ClassifierEngine::load_priors,  py::arg("priors"))
          .def("build_index", &ClassifierEngine::build_index,
               py::arg("max_depth") = ClassifierEngine::MAX_DEPTH,
               py::arg("min_log_prob") = ClassifierEngine::MIN_LOG_PROB)
          .def("save_index", &ClassifierEngine::save_index, py::arg("path"))
          .def("load_index", &ClassifierEngine::load_index, py::arg("path"))
          .def("classify", &ClassifierEngine::classify, py::arg("fen"), py::arg("top_n") = 5);
}