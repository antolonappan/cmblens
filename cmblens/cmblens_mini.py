import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String,select
import hashlib
import os

def hash_maps(maps):
    return hashlib.sha224(maps).hexdigest()

class MetaSIM:
    
    def __init__(self,fname,verbose=False):
        self.engine = create_engine(f'sqlite:///{fname}', echo=verbose)
        meta = MetaData()
        self.simulation = Table(
                           'simulation', meta, 
                           Column('id', Integer, primary_key = True), 
                           Column('seed', Integer), 
                           Column('hash_value', String),
                        )
        meta.create_all(self.engine)

    
    def get_row(self,idx):
        conn = self.engine.connect()
        sel = self.simulation.select().where(self.simulation.c.id==idx)
        l = conn.execute(sel).fetchall()
        conn.close()
        return l[0]

    
    def get_seed(self,idx):
        __,seed,__ = self.get_row(idx)
        return seed
        
    def get_hash(self,idx):
        __,__,hash_value = self.get_row(idx)
        return hash_value
    
    def checkhash(self,idx,hashv):
        return self.get_hash(idx) == hashv

class CMBLensed:
    """
    Lensing class:
    It saves seeds, Phi Map and Lensed CMB maps
    
    """
    def __init__(self,outfolder,nsim,scalar,with_tensor,lensed,do_tensor=False,verbose=False):
        self.outfolder = outfolder
        self.cl_unl = camb_clfile(scalar)
        self.cl_pot = camb_clfile(with_tensor)
        self.cl_len = camb_clfile(lensed)
        self.nside = 2048
        self.lmax = 4096
        self.verbose = verbose
        self.nsim = nsim

        
        self.cmb_dir = os.path.join(self.outfolder,f"CMB")
        self.mass_dir = os.path.join(self.outfolder,f"MASS") 
        

        self.meta = MetaSIM(os.path.join(self.outfolder,'META.db'),verbose)
        


    
    def vprint(self,string):
        if self.verbose:
            print(string)

    @property
    def get_kmap(self):
        fname = os.path.join(self.mass_dir,'kappa.fits')
        return hp.read_map(fname)

    @property
    def get_kappa(self):
        return hp.map2alm(self.get_kmap,lmax=4096)
    
    @property
    def get_phi(self):
        fname = os.path.join(self.mass_dir,'phi.fits')
        return hp.read_alm(fname)
        
    def plot_pp(self):
        data = hp.alm2cl(self.get_phi)
        theory = self.cl_pot['pp']
        lmax = min(len(data),len(theory))
        l = np.arange(lmax)
        w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi * 1e7
        
        plt.figure(figsize=(8,8))
        plt.loglog(data[:lmax]*w(l),label='WebSky')
        plt.loglog(theory[:lmax]*w(l),label='Fiducial')
        plt.xlabel('$L$',fontsize=20)
        plt.ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$  [$x10^7$]',fontsize=20)
        plt.xlim(2,None)
        plt.legend(fontsize=20)

    def get_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"sims_{idx:03d}.fits")
        self.vprint(f"CMB fields from cache: {idx}")
        maps = hp.read_map(fname,(0,1,2),dtype=np.float64)
        if self.meta.checkhash(idx,hash_maps(maps)):
            print("HASH CHECK: OK")
        else:
            print("HASH CHECK: FAILED")
        return maps
    
    def plot_lensed(self,idx):
        w = lambda ell :ell * (ell + 1) / (2. * np.pi)
        maps = self.get_lensed(idx)
        alms = hp.map2alm(maps)
        clss = hp.alm2cl(alms)
        lmax_d = len(clss[0])
        lmax_t = len(self.cl_len['tt'])
        l_d = np.arange(lmax_d)
        l_t = np.arange(lmax_t)
        plt.figure(figsize=(8,8))
        plt.loglog(clss[0]*w(l_d))
        plt.loglog(self.cl_len['tt']*w(l_t))
        plt.loglog(clss[1]*w(l_d))
        plt.loglog(self.cl_len['ee']*w(l_t))
        plt.loglog(clss[2]*w(l_d))
        plt.loglog(self.cl_len['bb']*w(l_t))
        plt.loglog(clss[3]*w(l_d))
        plt.loglog(self.cl_len['te']*w(l_t))
        plt.xlim(2,2000)
        plt.xlabel('$\ell$', fontsize=20)
        plt.ylabel('$C_\ell$', fontsize=20)
        
def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls, tensCls or ScalCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    with open(fname) as f:
        firstline = next(f)
    keys = [i.lower() for i in firstline.split(' ') if i.isalpha()][1:]
    cols = np.loadtxt(fname).transpose()

    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)

    cls = {k : np.zeros(lmax + 1, dtype=float) for k in keys}

    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)

    w = lambda ell :ell * (ell + 1) / (2. * np.pi)
    wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
    wptpe = lambda ell :np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi) 
    for i, k in enumerate(keys):
        if k == 'pp':
            we = wpp(ell)
        elif 'p' in k and ('e' in k or 't' in k):
            we = wptpe(ell)
        else:
            we = w(ell)
        cls[k][ell[idc]] = cols[i + 1][idc] / we[idc]
    return cls