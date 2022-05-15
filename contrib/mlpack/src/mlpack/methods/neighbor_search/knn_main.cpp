/**
 * @file methods/neighbor_search/knn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the kNN executable.  Allows some number of standard
 * options.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/math/random.hpp>

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME knn

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "neighbor_search.hpp"
#include "unmap.hpp"
#include "ns_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Convenience typedef.
typedef NSModel<NearestNeighborSort> KNNModel;

// Program Name.
BINDING_USER_NAME("k-Nearest-Neighbors Search");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of k-nearest-neighbor search using single-tree and "
    "dual-tree algorithms.  Given a set of reference points and query points, "
    "this can find the k nearest neighbors in the reference set of each query "
    "point using trees; trees that are built can be saved for future use.");

// Long description.
BINDING_LONG_DESC(
    "This program will calculate the k-nearest-neighbors of a set of "
    "points using kd-trees or cover trees (cover tree support is experimental "
    "and may be slow). You may specify a separate set of "
    "reference points and query points, or just a reference set which will be "
    "used as both the reference and query set.");

// Example.
BINDING_EXAMPLE(
    "For example, the following command will calculate the 5 nearest neighbors "
    "of each point in " + PRINT_DATASET("input") + " and store the distances "
    "in " + PRINT_DATASET("distances") + " and the neighbors in " +
    PRINT_DATASET("neighbors") + ": "
    "\n\n" +
    PRINT_CALL("knn", "k", 5, "reference", "input", "neighbors", "neighbors",
        "distances", "distances") +
    "\n\n"
    "The output is organized such that row i and column j in the neighbors "
    "output matrix corresponds to the index of the point in the reference set "
    "which is the j'th nearest neighbor from the point in the query set with "
    "index i.  Row j and column i in the distances output matrix corresponds to"
    " the distance between those two points.");

// See also...
BINDING_SEE_ALSO("@lsh", "#lsh");
BINDING_SEE_ALSO("@krann", "#krann");
BINDING_SEE_ALSO("@kfn", "#kfn");
BINDING_SEE_ALSO("NeighborSearch tutorial (k-nearest-neighbors)",
        "@doxygen/nstutorial.html");
BINDING_SEE_ALSO("Tree-independent dual-tree algorithms (pdf)",
        "http://proceedings.mlr.press/v28/curtin13.pdf");
BINDING_SEE_ALSO("mlpack::neighbor::NeighborSearch C++ class documentation",
        "@doxygen/classmlpack_1_1neighbor_1_1NeighborSearch.html");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");
PARAM_MATRIX_IN("true_distances", "Matrix of true distances to compute "
    "the effective error (average relative error) (it is printed when -v is "
    "specified).", "D");
PARAM_UMATRIX_IN("true_neighbors", "Matrix of true neighbors to compute the "
    "recall (it is printed when -v is specified).", "T");

// The option exists to load or save models.
PARAM_MODEL_IN(KNNModel, "input_model", "Pre-trained kNN model.", "m");
PARAM_MODEL_OUT(KNNModel, "output_model", "If specified, the kNN model will be "
    "output here.", "M");

// The user may specify a query file of query points and a number of nearest
// neighbors to search for.
PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");
PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);

// The user may specify the type of tree to use, and a few parameters for tree
// building.
PARAM_STRING_IN("tree_type", "Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', "
    "'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', "
    "'r-plus-plus', 'spill', 'oct'.", "t", "kd");
PARAM_INT_IN("leaf_size", "Leaf size for tree building (used for kd-trees, vp "
    "trees, random projection trees, UB trees, R trees, R* trees, X trees, "
    "Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees).", "l",
    20);
PARAM_DOUBLE_IN("tau", "Overlapping size (only valid for spill trees).", "u",
    0);
PARAM_DOUBLE_IN("rho", "Balance threshold (only valid for spill trees).", "b",
    0.7);

PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT_IN("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

// Search settings.
PARAM_STRING_IN("algorithm", "Type of neighbor search: 'naive', 'single_tree', "
    "'dual_tree', 'greedy'.", "a", "dual_tree");
PARAM_DOUBLE_IN("epsilon", "If specified, will do approximate nearest neighbor "
    "search with given relative error.", "e", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  RequireOnlyOnePassed(params, { "reference", "input_model" }, true);

  ReportIgnoredParam(params, {{ "input_model", true }}, "tree_type");
  ReportIgnoredParam(params, {{ "input_model", true }}, "random_basis");
  ReportIgnoredParam(params, {{ "input_model", true }}, "tau");
  ReportIgnoredParam(params, {{ "input_model", true }}, "rho");
  if (params.Has("input_model") && params.Has("leaf_size"))
  {
    Log::Warn << PRINT_PARAM_STRING("leaf_size") << " will only be considered"
        << " for the query tree, because --input_model_file is specified."
        << endl;
  }

  // The user should give something to do...
  RequireAtLeastOnePassed(params, { "k", "output_model" }, false,
      "no results will be saved");

  // If the user specifies k but no output files, they should be warned.
  if (params.Has("k"))
  {
    RequireAtLeastOnePassed(params, { "neighbors", "distances" }, false,
        "nearest neighbor search results will not be saved");
  }

  // If the user specifies output files but no k, they should be warned.
  ReportIgnoredParam(params, {{ "k", false }}, "neighbors");
  ReportIgnoredParam(params, {{ "k", false }}, "distances");
  ReportIgnoredParam(params, {{ "k", false }}, "true_neighbors");
  ReportIgnoredParam(params, {{ "k", false }}, "true_distances");
  ReportIgnoredParam(params, {{ "k", false }}, "query");

  // Sanity check on leaf size.
  RequireParamValue<int>(params, "leaf_size", [](int x) { return x > 0; },
      true, "leaf size must be positive");
  const int lsInt = params.Get<int>("leaf_size");

  // Sanity check on tau.
  RequireParamValue<double>(params, "tau", [](double x) { return x >= 0.0; },
      true, "tau must be positive");
  const double tau = params.Get<double>("tau");


  // Sanity check on rho.
  const double rho = params.Get<double>("rho");
  RequireParamValue<double>(params, "rho",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "rho must be in the range [0, 1]");
  if (params.Get<string>("tree_type") != "spill")
  {
    ReportIgnoredParam(params, "tau", "spill trees are not being used");
    ReportIgnoredParam(params, "rho", "spill trees are not being used");
  }

  // Sanity check on epsilon.
  const double epsilon = params.Get<double>("epsilon");
  RequireParamValue<double>(params, "epsilon",
      [](double x) { return x >= 0.0; }, true, "epsilon must be positive");

  // We either have to load the reference data, or we have to load the model.
  KNNModel* knn;

  const string algorithm = params.Get<string>("algorithm");
  RequireParamInSet<string>(params, "algorithm", { "naive", "single_tree",
      "dual_tree", "greedy" }, true, "unknown neighbor search algorithm");
  NeighborSearchMode searchMode = DUAL_TREE_MODE;

  if (algorithm == "naive")
    searchMode = NAIVE_MODE;
  else if (algorithm == "single_tree")
    searchMode = SINGLE_TREE_MODE;
  else if (algorithm == "dual_tree")
    searchMode = DUAL_TREE_MODE;
  else if (algorithm == "greedy")
    searchMode = GREEDY_SINGLE_TREE_MODE;

  if (params.Has("reference"))
  {
    // Get all the parameters.
    const string treeType = params.Get<string>("tree_type");
    const bool randomBasis = params.Has("random_basis");

    KNNModel::TreeTypes tree = KNNModel::KD_TREE;
    RequireParamInSet<string>(params, "tree_type", { "kd", "cover", "r",
        "r-star", "ball", "x", "hilbert-r", "r-plus", "r-plus-plus", "spill",
        "vp", "rp", "max-rp", "ub", "oct" }, true, "unknown tree type");

    knn = new KNNModel();

    if (treeType == "kd")
      tree = KNNModel::KD_TREE;
    else if (treeType == "cover")
      tree = KNNModel::COVER_TREE;
    else if (treeType == "r")
      tree = KNNModel::R_TREE;
    else if (treeType == "r-star")
      tree = KNNModel::R_STAR_TREE;
    else if (treeType == "ball")
      tree = KNNModel::BALL_TREE;
    else if (treeType == "x")
      tree = KNNModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = KNNModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = KNNModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = KNNModel::R_PLUS_PLUS_TREE;
    else if (treeType == "spill")
      tree = KNNModel::SPILL_TREE;
    else if (treeType == "vp")
      tree = KNNModel::VP_TREE;
    else if (treeType == "rp")
      tree = KNNModel::RP_TREE;
    else if (treeType == "max-rp")
      tree = KNNModel::MAX_RP_TREE;
    else if (treeType == "ub")
      tree = KNNModel::UB_TREE;
    else if (treeType == "oct")
      tree = KNNModel::OCTREE;

    knn->TreeType() = tree;
    knn->RandomBasis() = randomBasis;
    knn->LeafSize() = size_t(lsInt);
    knn->Tau() = tau;
    knn->Rho() = rho;

    Log::Info << "Using reference data from "
        << params.GetPrintable<arma::mat>("reference") << "." << endl;

    arma::mat referenceSet = std::move(params.Get<arma::mat>("reference"));

    knn->BuildModel(timers, std::move(referenceSet), searchMode, epsilon);
  }
  else
  {
    // Load the model from file.
    knn = params.Get<KNNModel*>("input_model");

    // Adjust search mode.
    knn->SearchMode() = searchMode;
    knn->Epsilon() = epsilon;

    // If leaf_size wasn't provided, let's consider the current value in the
    // loaded model.  Else, update it (only considered when building the query
    // tree).
    if (params.Has("leaf_size"))
      knn->LeafSize() = size_t(lsInt);

    Log::Info << "Loaded kNN model from '"
        << params.GetPrintable<KNNModel*>("input_model") << "' (trained on "
        << knn->Dataset().n_rows << "x" << knn->Dataset().n_cols
        << " dataset)." << endl;
  }

  // Perform search, if desired.
  if (params.Has("k"))
  {
    const size_t k = (size_t) params.Get<int>("k");

    arma::mat queryData;
    if (params.Has("query"))
    {
      Log::Info << "Using query data from "
          << params.GetPrintable<arma::mat>("query") << "." << endl;
      queryData = std::move(params.Get<arma::mat>("query"));
      if (queryData.n_rows != knn->Dataset().n_rows)
      {
        // Clean memory if needed before crashing.
        const size_t dimensions = knn->Dataset().n_rows;
        if (params.Has("reference"))
          delete knn;
        Log::Fatal << "Query has invalid dimensions(" << queryData.n_rows <<
            "); should be " << dimensions << "!" << endl;
      }
    }

    // Sanity check on k value: must be greater than 0, must be less than or
    // equal to the number of reference points.  Since it is unsigned,
    // we only test the upper bound.
    if (k > knn->Dataset().n_cols)
    {
      // Clean memory if needed before crashing.
      const size_t referencePoints = knn->Dataset().n_cols;
      if (params.Has("reference"))
        delete knn;
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less "
          << "than or equal to the number of reference points ("
          << referencePoints << ")." << endl;
    }

    // Sanity check on k value: must not be equal to the number of reference
    // points when query data has not been provided.
    if (!params.Has("query") && k == knn->Dataset().n_cols)
    {
      // Clean memory if needed before crashing.
      const size_t referencePoints = knn->Dataset().n_cols;
      if (params.Has("reference"))
        delete knn;
      Log::Fatal << "Invalid k: " << k << "; must be less than the number of "
          << "reference points (" << referencePoints << ") if query data has "
          << "not been provided." << endl;
    }

    // Now run the search.
    arma::Mat<size_t> neighbors;
    arma::mat distances;

    if (params.Has("query"))
      knn->Search(timers, std::move(queryData), k, neighbors, distances);
    else
      knn->Search(timers, k, neighbors, distances);

    Log::Info << "Search complete." << endl;

    // Calculate the effective error, if desired.
    if (params.Has("true_distances"))
    {
      if (knn->TreeType() != KNNModel::SPILL_TREE && knn->Epsilon() == 0)
        Log::Warn << PRINT_PARAM_STRING("true_distances") << "specified, but "
            << "the search is exact, so there is no need to calculate the "
            << "error!" << endl;

      arma::mat trueDistances =
          std::move(params.Get<arma::mat>("true_distances"));

      if (trueDistances.n_rows != distances.n_rows ||
          trueDistances.n_cols != distances.n_cols)
      {
        if (params.Has("reference"))
          delete knn;
        Log::Fatal << "The true distances file must have the same number of "
            << "values than the set of distances being queried!" << endl;
      }

      Log::Info << "Effective error: " << KNN::EffectiveError(distances,
          trueDistances) << endl;
    }

    // Calculate the recall, if desired.
    if (params.Has("true_neighbors"))
    {
      if (knn->TreeType() != KNNModel::SPILL_TREE && knn->Epsilon() == 0)
        Log::Warn << PRINT_PARAM_STRING("true_neighbors") << " specified, but "
            << " the search is exact, so there is no need to calculate the "
            << "recall!" << endl;

      arma::Mat<size_t> trueNeighbors =
          std::move(params.Get<arma::Mat<size_t>>("true_neighbors"));

      if (trueNeighbors.n_rows != neighbors.n_rows ||
          trueNeighbors.n_cols != neighbors.n_cols)
      {
        if (params.Has("reference"))
          delete knn;
        Log::Fatal << "The true neighbors file must have the same number of "
            << "values than the set of neighbors being queried!" << endl;
      }

      Log::Info << "Recall: " << KNN::Recall(neighbors, trueNeighbors) << endl;
    }

    // Save output.
    params.Get<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    params.Get<arma::mat>("distances") = std::move(distances);
  }

  params.Get<KNNModel*>("output_model") = knn;
}
