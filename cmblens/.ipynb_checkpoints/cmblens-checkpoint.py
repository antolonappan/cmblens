class CMBLensed:
    """
    Lensing class:
    It saves seeds, Phi Map and Lensed CMB maps
    
    """
    def __init__(self,outfolder,nsim,cl_path,scal_file,pot_file,len_file,sim_set,verbose=False):
        self.outfolder = outfolder
        self.cl_unl = camb_clfile2(os.path.join(cl_path, scal_file))
        self.cl_pot = camb_clfile2(os.path.join(cl_path, pot_file))
        self.cl_len = camb_clfile2(os.path.join(cl_path, len_file))
        self.nside = 2048
        self.lmax = 4096
        self.dlmax = 1024
        self.facres = 0
        self.verbose = verbose
        self.nsim = nsim
        self.sim_set = sim_set
        
        if sim_set == 1:
            mass_set = 1
        elif (sim_set == 2) or (sim_set == 3):
            mass_set = 2
        elif sim_set == 4:
            assert len_file is not None
            mass_set = 1
        else:
            raise ValueError
        
        self.mass_set = mass_set
        
        #folder for CMB
        self.cmb_dir = os.path.join(self.outfolder,f"CMB_SET{self.sim_set}")
        #folder for mass
        self.mass_dir = os.path.join(self.outfolder,f"MASS_SET{self.mass_set}") 
        
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            os.makedirs(self.mass_dir,exist_ok=True) 
            os.makedirs(self.cmb_dir,exist_ok=True)
        
        
        fname = os.path.join(self.outfolder,'seeds.pkl')
        if (not os.path.isfile(fname)) and (mpi.rank == 0):
            seeds = self.get_seeds
            pk.dump(seeds, open(fname,'wb'), protocol=2)
        mpi.barrier()
        self.seeds = pk.load(open(fname,'rb'))
        
        
        # Here I saves a dictonary with the artibutes of this class and given Cls. 
        # So everytime when this instance run it checks for the same setup
        # If any artribute has changed from the previous run
        fnhash = os.path.join(self.outfolder, "lensing_sim_hash.pk")
        if (mpi.rank == 0) and (not os.path.isfile(fnhash)):
            pk.dump(self.hashdict(), open(fnhash, 'wb'), protocol=2)
        mpi.barrier()
        
        hash_check(pk.load(open(fnhash, 'rb')), self.hashdict())

    def hashdict(self):
        return {'nside':self.nside,
                'lmax':self.lmax,
                'cl_ee': clhash(self.cl_unl['ee']),
                'cl_pp': clhash(self.cl_pot['pp']),
                'cl_tt': clhash(self.cl_len['tt']),
               }
    @property
    def get_seeds(self):
        """
        non-repeating seeds
        """
        seeds =[]
        no = 0
        while no <= self.nsim-1:
            r = np.random.randint(11111,99999)
            if r not in seeds:
                seeds.append(r)
                no+=1
        return seeds
    
    def vprint(self,string):
        if self.verbose:
            print(string)
                  
    def get_phi(self,idx):
        """
        set a seed
        generate phi_LM
        Save the phi
        """
        fname = os.path.join(self.mass_dir,f"phi_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"Phi field from cache: {idx}")
            return hp.read_alm(fname)
        else:
            rNo = self.mass_set - 1
            np.random.seed(self.seeds[idx]-rNo)
            plm = hp.synalm(self.cl_pot['pp'], lmax=self.lmax + self.dlmax, new=True)
            hp.write_alm(fname,plm)
            self.vprint(f"Phi field cached: {idx}")
            return plm
        
    def get_kappa(self,idx):
        """
        generate deflection field
        sqrt(L(L+1)) * \phi_{LM}
        """
        der = np.sqrt(np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2))
        return hp.almxfl(self.get_phi(idx), der)
    
    def get_unlensed_alm(self,idx):
        self.vprint(f"Synalm-ing the Unlensed CMB temp: {idx}")
        Cls = [self.cl_unl['tt'],self.cl_unl['ee'],self.cl_unl['tt']*0,self.cl_unl['te']]
        np.random.seed(self.seeds[idx]+self.sim_set)
        alms = hp.synalm(Cls,lmax=self.lmax + self.dlmax,new=True)
        return alms
    
    def get_gauss_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"cmb_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"CMB Gaussian fields from cache: {idx}")
            return hp.read_map(fname,(0,1,2),dtype=np.float64)
        else:
            Cls = [self.cl_len['tt'],self.cl_len['ee'],self.cl_len['bb'],self.cl_len['te']]
            np.random.seed(self.seeds[idx])
            maps = hp.synfast(Cls,self.nside,self.lmax,pol=True)
            hp.write_map(fname,maps,dtype=np.float64)
            self.vprint(f"CMB Gaussian fields cached: {idx}")
            return maps
            
            

    
    def get_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"cmb_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"CMB fields from cache: {idx}")
            return hp.read_map(fname,(0,1,2),dtype=np.float64)
        else:
            dlm = self.get_kappa(idx)
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
            hp.write_map(fname,[T,Q,U],dtype=np.float64)
            self.vprint(f"CMB field cached: {idx}")         
            return [T,Q,U]
        
        
    def run_job(self):
        jobs = np.arange(self.nsim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Lensing map-{i} in processor-{mpi.rank}")
            if self.sim_set == 4:
                NULL = self.get_gauss_lensed(i)
            else:
                NULL = self.get_lensed(i)
            del NULL