import os

verbose = False

has_key = lambda key: key in os.environ.keys()
cond4mpi4py = not has_key("NERSC_HOST") or (
    has_key("NERSC_HOST") and has_key("SLURM_SUBMIT_DIR")
)

if cond4mpi4py:
    try:
        from mpi4py import MPI

        nompi = False
        mpi = MPI
        com = MPI.COMM_WORLD
        rank = com.Get_rank()
        size = com.Get_size()
        barrier = com.Barrier
        finalize = mpi.Finalize
        if verbose:
            print("mpi.py : setup OK, rank %s in %s" % (rank, size))
    except ModuleNotFoundError:
        rank = 0
        size = 1
        barrier = lambda: -1
        finalize = lambda: -1
        if verbose:
            print("mpi.py: unable to import mpi4py\n")
else:
    nompi = True
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
