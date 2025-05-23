// ============================================================================
// File: main.cpp (Assignment 3 – Parallel Quick‑Sort with MPI)
// ----------------------------------------------------------------------------
//  * Builds a single executable named `quicksort` (see Makefile below).
//  * Implements all functions declared in the supplied quicksort.h and
//    pivot.h headers.
//  * Fulfils every specification in Assignment‑3 (April 22 2025).
// ----------------------------------------------------------------------------
//  Compilation (via provided Makefile):
//      $ module load gcc openmpi/5.0.5   # on UPPMAX or equivalent
//      $ make            # creates ./quicksort
//  Usage:
//      mpirun -np 4 ./quicksort <input_file> <output_file> <pivot_strategy>
//  where <pivot_strategy> is 1 (median‑in‑root), 2 (mean‑of‑medians) or
//  3 (median‑of‑medians).
// ============================================================================

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include "quicksort.h"
#include "pivot.h"

// ‑‑‑‑ Local helpers ---------------------------------------------------------
static int min_int(int a,int b){ return a<b?a:b; }

// ==========================================================================
//                       Implementation of pivot.h
// ==========================================================================

int compare(const void *v1,const void *v2){
	int a=*static_cast<const int*>(v1);
	int b=*static_cast<const int*>(v2);
	return (a>b)-(a<b); // returns +1, 0 or ‑1
}

int get_larger_index(int *elements,int n,int val){
	// Binary search – first element > val (upper bound)
	int lo=0,hi=n; // hi is one‑past‑end
	while(lo<hi){
		int mid=(lo+hi)/2;
		if(elements[mid]<=val) lo=mid+1; else hi=mid;
	}
	return lo; // in range [0,n]
}

int get_median(int *elements,int n){
	if(n==0) return 0; // undefined – caller ensures n>0 in normal use
	if(n&1) return elements[n/2];
	// Even – average the middle pair (assignment says numbers are ints)
	int a=elements[n/2-1];
	int b=elements[n/2];
	return a/2 + b/2 + ( (a&1)&(b&1) ); // safe average without overflow
}

// Forward declarations of private helpers
static int broadcast_pivot(int &pivot,MPI_Comm comm);

int select_pivot_median_root(int *elements,int n,MPI_Comm comm){
	int rank; MPI_Comm_rank(comm,&rank);
	int pivot=0;
	if(rank==ROOT){
		pivot=get_median(elements,n);
	}
	broadcast_pivot(pivot,comm);
	// Return partition index for *this* process
	return get_larger_index(elements,n,pivot);
}

int select_pivot_mean_median(int *elements,int n,MPI_Comm comm){
	int local_med=get_median(elements,n);
	long long sum=0;
	long long local=local_med;
	MPI_Allreduce(&local,&sum,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
	int size; MPI_Comm_size(comm,&size);
	int pivot=static_cast<int>(sum/size);
	broadcast_pivot(pivot,comm);
	return get_larger_index(elements,n,pivot);
}

int select_pivot_median_median(int *elements,int n,MPI_Comm comm){
	int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
	int local_med=get_median(elements,n);
	std::vector<int> all(size);
	MPI_Gather(&local_med,1,MPI_INT,all.data(),1,MPI_INT,ROOT,comm);
	int pivot=0;
	if(rank==ROOT){
		std::sort(all.begin(),all.end());
		pivot = (size&1)? all[size/2] : (all[size/2-1]+all[size/2])/2;
	}
	broadcast_pivot(pivot,comm);
	return get_larger_index(elements,n,pivot);
}

int select_pivot_smallest_root(int *elements,int n,MPI_Comm comm){
	int rank; MPI_Comm_rank(comm,&rank);
	int pivot=0;
	if(rank==ROOT && n>0) pivot=elements[0];
	broadcast_pivot(pivot,comm);
	return get_larger_index(elements,n,pivot);
}

int select_pivot(int pivot_strategy,int *elements,int n,MPI_Comm comm){
	switch(pivot_strategy){
		case MEDIAN_ROOT:   return select_pivot_median_root(elements,n,comm);
		case MEAN_MEDIAN:   return select_pivot_mean_median(elements,n,comm);
		case MEDIAN_MEDIAN: return select_pivot_median_median(elements,n,comm);
		default:            return select_pivot_smallest_root(elements,n,comm);
	}
}

static int broadcast_pivot(int &pivot,MPI_Comm comm){
	MPI_Bcast(&pivot,1,MPI_INT,ROOT,comm);
	return pivot;
}

// ==========================================================================
//                       Implementation of quicksort.h
// ==========================================================================

int read_input(char *file_name,int **elements){
	FILE *f=fopen(file_name,"r");
	if(!f){ perror("fopen input"); MPI_Abort(MPI_COMM_WORLD,2);}
	int n; if(fscanf(f,"%d",&n)!=1){ fprintf(stderr,"Invalid input file\n"); MPI_Abort(MPI_COMM_WORLD,3);}
	*elements=(int*)malloc(sizeof(int)*n);
	for(int i=0;i<n;++i){ if(fscanf(f,"%d",(*elements)+i)!=1){ fprintf(stderr,"Unexpected EOF\n"); MPI_Abort(MPI_COMM_WORLD,4);} }
	fclose(f);
	return n;
}

void swap(int *e1,int *e2){ int tmp=*e1; *e1=*e2; *e2=tmp; }

int sorted_ascending(int *elements,int n){
	for(int i=1;i<n;++i) if(elements[i-1]>elements[i]) return 0;
	return 1;
}

void merge_ascending(int *v1,int n1,int *v2,int n2,int *result){
	int i=0,j=0,k=0;
	while(i<n1 && j<n2){
		if(v1[i]<=v2[j]) result[k++]=v1[i++]; else result[k++]=v2[j++];
	}
	while(i<n1) result[k++]=v1[i++];
	while(j<n2) result[k++]=v2[j++];
}

int check_and_print(int *elements,int n,char *file_name){
	if(!sorted_ascending(elements,n)){
		fprintf(stdout,"Elements NOT sorted – first disorder at index i where a[i] > a[i+1]\n");
	}
	FILE *f=fopen(file_name,"w");
	if(!f){ perror("fopen output"); return -2; }
	for(int i=0;i<n;++i) fprintf(f,"%d%s",elements[i],(i==n-1)?"":" ");
	fclose(f);
	return 0;
}

int distribute_from_root(int *all_elements,int n,int **my_elements){
	int size,rank; MPI_Comm_size(MPI_COMM_WORLD,&size); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	// counts and displs
	std::vector<int> counts(size),displs(size);
	int base=n/size,rem=n%size;
	for(int i=0;i<size;++i){ counts[i]=base+(i<rem); }
	displs[0]=0; for(int i=1;i<size;++i) displs[i]=displs[i-1]+counts[i-1];

	*my_elements=(int*)malloc(sizeof(int)*counts[rank]);
	MPI_Scatterv(all_elements,counts.data(),displs.data(),MPI_INT,
				 *my_elements,counts[rank],MPI_INT,ROOT,MPI_COMM_WORLD);
	return counts[rank];
}

void gather_on_root(int *all_elements,int *my_elements,int local_n){
	int size,rank; MPI_Comm_size(MPI_COMM_WORLD,&size); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	std::vector<int> counts(size);
	MPI_Gather(&local_n,1,MPI_INT,counts.data(),1,MPI_INT,ROOT,MPI_COMM_WORLD);
	std::vector<int> displs(size);
	if(rank==ROOT){ displs[0]=0; for(int i=1;i<size;++i) displs[i]=displs[i-1]+counts[i-1]; }
	MPI_Gatherv(my_elements,local_n,MPI_INT,all_elements,counts.data(),displs.data(),MPI_INT,ROOT,MPI_COMM_WORLD);
}

// -------------------- global_sort -----------------------------------------
static int recursive_global_sort(int **elements,int n,MPI_Comm comm,int pivot_strategy){
	int size; MPI_Comm_size(comm,&size);
	if(size==1) return n; // done

	int rank; MPI_Comm_rank(comm,&rank);
	int half=size/2;

	// Select pivot & partition
	int pivot_idx=select_pivot(pivot_strategy,*elements,n,comm);
	int *lo=*elements;          int n_lo=pivot_idx;
	int *hi=*elements+pivot_idx;int n_hi=n-pivot_idx;

	bool keep_low = (rank < half); // colour 0 keeps low part
	int partner   = keep_low ? rank+half : rank-half;

	// Exchange sizes
	int send_cnt = keep_low ? n_hi : n_lo;
	int recv_cnt = 0;
	MPI_Sendrecv(&send_cnt,1,MPI_INT,partner,0,&recv_cnt,1,MPI_INT,partner,0,comm,MPI_STATUS_IGNORE);

	// Allocate buffers
	int *send_buf = keep_low ? hi : lo;
	std::vector<int> recv_buf(recv_cnt);
	MPI_Sendrecv(send_buf,send_cnt,MPI_INT,partner,1,recv_buf.data(),recv_cnt,MPI_INT,partner,1,comm,MPI_STATUS_IGNORE);

	// Build new local array (kept part + received part)
	int new_keep_cnt = keep_low ? n_lo : n_hi;
	int *keep_buf = keep_low ? lo : hi;
	int new_n = new_keep_cnt + recv_cnt;
	int *merged = (int*)malloc(sizeof(int)*new_n);
	merge_ascending(keep_buf,new_keep_cnt,recv_buf.data(),recv_cnt,merged);

	free(*elements);
	*elements = merged;

	// Split communicator and recurse
	int color = keep_low?0:1;
	MPI_Comm subcomm; MPI_Comm_split(comm,color,rank,&subcomm);
	int final_n = recursive_global_sort(elements,new_n,subcomm,pivot_strategy);
	MPI_Comm_free(&subcomm);
	return final_n;
}

int global_sort(int **elements,int n,MPI_Comm comm,int pivot_strategy){
	return recursive_global_sort(elements,n,comm,pivot_strategy);
}

// ==========================================================================
//                              main()
// ==========================================================================

int main(int argc,char **argv){
	MPI_Init(&argc,&argv);

	if(argc!=4){
		if(argc>1){ int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank); if(rank==ROOT) fprintf(stderr,"Usage: %s <input> <output> <pivot_strategy>\n",argv[0]); }
		MPI_Abort(MPI_COMM_WORLD,1);
	}
	char *input_file = argv[1];
	char *output_file= argv[2];
	int pivot_strategy = atoi(argv[3]);

	int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	int *all_elements=nullptr; int n=0;
	if(rank==ROOT){ n=read_input(input_file,&all_elements); }

	// Distribute
	int *my_elements=nullptr; int local_n = distribute_from_root(all_elements,n,&my_elements);

	// Local sort (not timed yet)
	double t0,t1; MPI_Barrier(MPI_COMM_WORLD); t0=MPI_Wtime();
	qsort(my_elements,local_n,sizeof(int),compare);

	// Global sort (recursive)
	int new_local_n = global_sort(&my_elements,local_n,MPI_COMM_WORLD,pivot_strategy);
	MPI_Barrier(MPI_COMM_WORLD); t1=MPI_Wtime();

	if(rank==ROOT) printf("%.6f\n",t1-t0);

	// Gather & write output
/* ---------- gather the (already locally‑sorted) chunks on rank 0 ---------- */
if (rank == ROOT) {
    free(all_elements);
    all_elements = static_cast<int*>(malloc(sizeof(int) * n));
}
gather_on_root(all_elements, my_elements, new_local_n);
	if (rank == ROOT) {
		check_and_print(all_elements, n, output_file);
		free(all_elements);
	}
/* ---------- tidy up ------------------------------------------------------ */
free(my_elements);
MPI_Finalize();
return 0;}

// ============================================================================
//                               Makefile
// ============================================================================
// (Put the following into a separate file named `Makefile` inside the A3 folder)
//
// CC       = mpic++
// CFLAGS   = -O3 -Wall -Wextra -std=c++17
// TARGET   = quicksort
// SOURCES  = main.cpp
//
// all: $(TARGET)
//
// $(TARGET): $(SOURCES) pivot.h quicksort.h
// 	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)
//
// clean:
// 	rm -f $(TARGET) *.o
//
// ============================================================================
