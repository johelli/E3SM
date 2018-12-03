#ifndef ZOLTANINTERFACEHPP
#define ZOLTANINTERFACEHPP

#if TRILINOS_HAVE_ZOLTAN2
#include <Zoltan2_XpetraCrsGraphAdapter.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_TaskMapping.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Map.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <Zoltan2_MappingProblem.hpp>
#include <Zoltan2_MappingSolution.hpp>
#include <Zoltan2_EvaluatePartition.hpp>
#include <Zoltan2_EvaluateMapping.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#endif

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

  void sort_graph(
    int     *nelem,
    int     *xadj,
    int     *adjncy,
    double  *adjwgt,
    double  *vwgt);

  void zoltan_partition_problem(
     int      *nelem,
     int      *xadj,
     int      *adjncy,
     double   *adjwgt,
     double   *vwgt,
     int      *nparts,
     MPI_Comm comm,
     double   *xcoord,
     double   *ycoord,
     double   *zcoord, 
     int      *coord_dimension,
     int      *result_parts,
     int      *partmethod,
     int      *mapmethod);

 void zoltan_map_problem(
     int      *nelem,
     int      *xadj,
     int      *adjncy,
     double   *adjwgt,
     double   *vwgt,
     int      *nparts,
     MPI_Comm comm,
     double   *xcoord,
     double   *ycoord,
     double   *zcoord, 
     int      *coord_dimension,
     int      *result_parts,
     int      *partmethod,
     int      *mapmethod);

  void zoltan_mapping_problem(
     int      *nelem,
     int      *xadj,
     int      *adjncy,
     double   *adjwgt,
     double   *vwgt,
     int      *nparts,
     MPI_Comm comm,
     double   *xcoord,
     double   *ycoord,
     double   *zcoord, 
     int      *coord_dimension,
     int      *result_parts,
     int      *partmethod,
     int      *mapmethod);

  void zoltan2_print_metrics(
     int      *nelem,
     int      *xadj,
     int      *adjncy,
     double   *adjwgt,
     double   *vwgt,
     int      *nparts,
     MPI_Comm comm,
     int      *result_parts);

  void zoltan2_print_metrics2(
     int      *nelem,
     int      *xadj,
     int      *adjncy,
     double   *adjwgt,
     double   *vwgt,
     int      *nparts,
     MPI_Comm comm,
     int      *result_parts);

#ifdef __cplusplus
}
#endif


