import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import sys
from cmblens import mpi
from cmblens.utils import camb_clfile,hash_maps,MetaSIM
import lenspyx


class CMBLensed:
    """
    Lensing class:
    It saves seeds, Phi Map and Lensed CMB maps
    
    """
    def __init__(self,outfolder,nsim,scalar,with_tensor,lensed,do_tensor=False,verbose=False,idx_start=0):
        self.outfolder = outfolder
        self.cl_unl = camb_clfile(scalar)
        self.cl_pot = camb_clfile(with_tensor)
        self.cl_len = camb_clfile(lensed)
        self.nside = 2048
        self.lmax = 4096
        self.dlmax = 1024
        self.facres = 0
        self.verbose = verbose
        self.nsim = nsim
        self.idx_start = idx_start

        
        self.cmb_dir = os.path.join(self.outfolder,f"CMB")
        self.mass_dir = os.path.join(self.outfolder,f"MASS") 
        
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            os.makedirs(self.mass_dir,exist_ok=True)
            os.makedirs(self.cmb_dir,exist_ok=True)
        

        self.meta = MetaSIM(os.path.join(self.outfolder,'META.db'),verbose)
        
        if mpi.rank == 0:
            seeds = self.meta.get_nseeds(self.idx_start+nsim)[self.idx_start:]
        else:
            seeds = None
        
        if mpi.nompi:
            self.seeds = seeds
        else:
            self.seeds = mpi.com.bcast(seeds,root=0)
        mpi.barrier()

    
    def vprint(self,string):
        if self.verbose:
            print(string)

    @property
    def get_kmap(self):
        fname = os.path.join(self.mass_dir,'kappa.fits')
        if os.path.isfile(fname):
            return hp.read_map(fname)
        else:
            self.vprint('Downloading kappa map')
            import requests
            kappa_url = 'https://mocks.cita.utoronto.ca/data/websky/v0.0/kap.fits'
            r = requests.get(kappa_url)

            with open(fname,'wb') as f:
                f.write(r.content)
            return hp.read_map(fname)

    
    @property
    def get_kappa(self):
        return hp.map2alm(self.get_kmap,lmax=2048)
    
    
    def get_phi(self,fid=False):
        """

        """
        name = 'phi_fid.fits' if fid else 'phi.fits'
        fname = os.path.join(self.mass_dir,name)
        if os.path.isfile(fname):
            return hp.read_alm(fname)
        else:
            fac = np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2)
            if fid:
                phi = hp.synalm(self.cl_pot['pp'], lmax=self.lmax + self.dlmax, new=True)
            else:
                phi = hp.almxfl(self.get_kappa,2/fac)
            hp.write_alm(fname,phi)
            return phi
           
    def get_deflection(self,fid=False):
        """
        generate deflection field
        sqrt(L(L+1)) * \phi_{LM}
        """
        der = np.sqrt(np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2))
        defl = hp.almxfl(self.get_phi(fid), der)
        defl[0] = 0
        return defl
        
    
    def plot_pp(self,fid=False):
        data = hp.alm2cl(self.get_phi(fid))
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
    
    def get_unlensed_alm(self,idx):
        self.vprint(f"Synalm-ing the Unlensed CMB temp: {idx}")
        Cls = [self.cl_unl['tt'],self.cl_unl['ee'],self.cl_unl['tt']*0,self.cl_unl['te']]
        np.random.seed(self.seeds[idx])
        alms = hp.synalm(Cls,lmax=self.lmax + self.dlmax,new=True)
        return alms

    def get_lensed(self,idx,fid=False):
        name = f"sims_{idx:03d}.fits" if not fid else f"sims_fid_{idx:03d}.fits"
        fname = os.path.join(self.cmb_dir,name)
        if os.path.isfile(fname):
            self.vprint(f"CMB fields from cache: {idx}")
            maps = hp.read_map(fname,(0,1,2),dtype=np.float64)
            if self.meta.checkhash(idx,hash_maps(maps)):
                print("HASH CHECK: OK")
            else:
                print("HASH CHECK: FAILED")
            return maps
        else:
            dlm = self.get_deflection(fid)
            Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], self.nside, 1, hp.Alm.getlmax(dlm.size))
            del dlm
            tlm,elm,blm = self.get_unlensed_alm(idx)
            del blm
            T  = lenspyx.alm2lenmap(tlm, [Red, Imd], self.nside, 
                                    facres=self.facres, 
                                    verbose=False)
            del tlm
            Q, U  = lenspyx.alm2lenmap_spin([elm, None],[Red, Imd], 
                                            self.nside, 2, facres=self.facres,
                                            verbose=False)
            del (Red, Imd, elm)
            maps = np.array([T,Q,U])
            hp.write_map(fname,maps,dtype=np.float64)
            self.vprint(f"CMB field cached: {idx}")
            mpi.barrier()
            self.meta.insert_hash_mpi(idx,hash_maps(maps))
            return maps
    
    def plot_lensed(self,idx,fid=False):
        w = lambda ell :ell * (ell + 1) / (2. * np.pi)
        maps = self.get_lensed(idx,fid)
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
        
    def run_job(self):
        jobs = np.arange(self.nsim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Lensing map-{i} in processor-{mpi.rank}")
            NULL = self.get_lensed(i)
    def save_unlensed(self):
        jobs = np.arange(self.nsim)
        for idx in jobs[mpi.rank::mpi.size]:
            print(f"Saving unlensed alms-{idx} in processor-{mpi.rank}")
            name = f"sims_unl_{idx:03d}.fits" if not fid else f"sims_unl_fid_{idx:03d}.fits"
            fname = os.path.join(self.cmb_dir,name)
            hp.write_alm(fname,self.get_unlensed_alm(i))
            
            

if __name__ == '__main__':
    base_dir = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/websky_extragal/websky_lensed_cmb'
    camb_dir = os.path.join(base_dir,'CAMB')
    scalar_file = os.path.join(camb_dir,'BBSims_scal_dls.dat')
    total_file = os.path.join(camb_dir,'BBSims_lenspotential.dat')
    lensed_file = os.path.join(camb_dir,'BBSims_lensed_dls.dat')
    idx_start = 50
    
    nsim = 150
    
    c = CMBLensed(base_dir,nsim,scalar_file,total_file,lensed_file,idx_start=idx_start)
    
    c.run_job()
    #c.save_unlensed()