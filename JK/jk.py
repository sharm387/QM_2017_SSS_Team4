import numpy as np
import psi4


def jk(mol,C,D,nel):
    # Get basital basis from a wavefunction
    # bas = wfn.basisset()
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    #build the complementary JKFIT basis for the bais set say aug-cc-pVDZ basis
    aux = psi4.core.BasisSet.build(mol,fitrole="JKFIT", other = "aug-cc-pVDZ")
    # get zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
    # Build instance of MintHelper
    mints = psi4.core.MintsHelper(bas)
    # Build (P|pq) raw 3-index ERIs, dimension( 1, Naux, nbf, nbf)
    Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
    Qls_tilde = np.squeeze(Qls_tilde) #remove 1-D
    # Build & invert COulomb metric, dimension(1,Naux,1,Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5,1.e-14)
    metric = np.squeeze(metric) 
    #Build PLs
    Pls = np.einsum("pq,qls->pls", metric, Qls_tilde)
    #compute Xp
    Xp = np.einsum("pls,ls->p", Pls,D)
    #compute Coulomb integral (J)
    J= np.einsum("lsp,p->ls", Pls.transpose(1,2,0), Xp)
    #exchange integral matrix
    eita_1 = np.einsum("lmP,pn->Pmp", Pls.transpose(1,2,0),C[:,:nel])
    eita_2 = np.einsum("Pnl,pm->Pnp", Pls,C[:,:5])
    K = np.einsum("Pmp,Pnp->mn",eita_1,eita_2)
    return J, K
    
