import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select
import hashlib
import itertools
from . import mpi
import os


def hash_maps(maps):
    return hashlib.sha224(maps).hexdigest()


class MetaSIM:
    def __init__(self, fname, verbose=False):
        if (mpi.rank == 0) and (not os.path.isfile(fname)):
            _ = create_engine(f"sqlite:///{fname}", echo=True)
        mpi.barrier()

        self.engine = create_engine(f"sqlite:///{fname}", echo=verbose)
        meta = MetaData()
        self.simulation = Table(
            "simulation",
            meta,
            Column("id", Integer, primary_key=True),
            Column("seed", Integer),
            Column("hash_value", String),
        )
        meta.create_all(self.engine)

    def insert_seed(self, idx, seed):
        conn = self.engine.connect()
        ins = self.simulation.insert().values(id=idx, seed=seed)
        conn.execute(ins)
        conn.close()

    def insert_seed_arr(self, seeds):
        for i, seed in enumerate(seeds):
            self.insert_seed(i, seed)

    def insert_hash_mpi(self, idx, hash_value):
        if mpi.rank != 0:
            req = mpi.com.isend(hash_value, dest=0, tag=mpi.rank)
            req.wait()
        else:
            for i in range(mpi.size):
                if i == 0:
                    self.insert_hash(i, hash_value)
                else:
                    req = mpi.com.irecv(source=i, tag=i)
                    data = req.wait()
                    self.insert_hash(i, data)
        mpi.barrier()

    def insert_hash(self, idx, hash_value):
        conn = self.engine.connect()
        upd = (
            self.simulation.update()
            .where(self.simulation.c.id == idx)
            .values(hash_value=hash_value)
        )
        conn.execute(upd)
        conn.close()

    def get_row(self, idx):
        conn = self.engine.connect()
        sel = self.simulation.select().where(self.simulation.c.id == idx)
        l = conn.execute(sel).fetchall()
        conn.close()
        try:
            return l[0]
        except IndexError:
            print("suspect an MPI error")
            return (idx, None, "0")

    def get_allseeds(self):
        conn = self.engine.connect()
        sel = select([self.simulation.c.seed])
        l = conn.execute(sel).fetchall()
        conn.close()
        seeds = list(itertools.chain(*l))
        if None in seeds:
            raise ValueError
        return seeds

    def get_seed(self, idx):
        try:
            __, seed, __ = self.get_row(idx)
            return seed
        except IndexError:
            r = self.__get_rand_seed__
            self.insert_seed(idx, r)
            return r

    def get_nseeds(self, nsims):
        seeds = []
        for i in range(nsims):
            seeds.append(self.get_seed(i))

        return seeds

    def get_hash(self, idx):
        __, __, hash_value = self.get_row(idx)
        return hash_value

    def checkhash(self, idx, hashv):
        return self.get_hash(idx) == hashv

    @property
    def __get_rand_seed__(self):
        while True:
            r = np.random.randint(11111, 99999)
            if r not in self.get_allseeds():
                return r


def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls, tensCls or ScalCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    with open(fname) as f:
        firstline = next(f)
    keys = [i.lower() for i in firstline.split(" ") if i.isalpha()][1:]
    cols = np.loadtxt(fname).transpose()

    ell = np.int_(cols[0])
    if lmax is None:
        lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)

    cls = {k: np.zeros(lmax + 1, dtype=float) for k in keys}

    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)

    w = lambda ell: ell * (ell + 1) / (2.0 * np.pi)
    wpp = lambda ell: ell**2 * (ell + 1) ** 2 / (2.0 * np.pi)
    wptpe = lambda ell: np.sqrt(ell.astype(float) ** 3 * (ell + 1.0) ** 3) / (
        2.0 * np.pi
    )
    for i, k in enumerate(keys):
        if k == "pp":
            we = wpp(ell)
        elif "p" in k and ("e" in k or "t" in k):
            we = wptpe(ell)
        else:
            we = w(ell)
        cls[k][ell[idc]] = cols[i + 1][idc] / we[idc]
    return cls
