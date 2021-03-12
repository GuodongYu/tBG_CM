from tBG.crystal.structures import CommensuStruct
from tBG.crystal.brillouin_zones import BZHexagonal
from tBG.utils import cart2frac, rotate_on_vec, frac2cart
import numpy as np
from tBG.crystal.structures import Graphene
from hankel import HankelTransform
from scipy.linalg.lapack import zheev
import time

"""
repeat the small twisted angle part in PRB 93, 035452(2016)
"""

### get K and RK ###
def get_K_and_RK(latt_vec_unrott, latt_vec_rott, ax=None):
    """
    return K, -K, RK -RK, g1, g2 
    K and -K are K and K_prim of the unrotted layer
    RK and -RK are the K and K_prim of the rotted layer
    g1 and g2 are the reciprocal lattice vectors of the Moire supercell
    """
    def get_K_Gs(latt_vec):
        bz = BZHexagonal(latt_vec)
        K = bz.special_points()['K'][0]
        G1 = -bz.recip_latt_vec[0]
        G2 = G1 - bz.recip_latt_vec[1]
        return bz, K, G1, G2
    bz_unrott, K, G1, G2 = get_K_Gs(latt_vec_unrott)
    bz_rott, RK, RG1, RG2 = get_K_Gs(latt_vec_rott)
    g1 = G1 - RG1
    g2 = G2 - RG2
    if ax: 
        bz_unrott._plot(ax, color='black')
        bz_rott._plot(ax, color='red')
        ax.scatter(K[0], K[1], color='black', marker='*')
        ax.scatter(RK[0], RK[1], color='red', marker='*')
        ax.plot([K[0],K[0]+G1[0]],[K[1],K[1]+G1[1]], color='black',ls='dashed')
        ax.plot([K[0],K[0]+G2[0]],[K[1],K[1]+G2[1]], color='black',ls='dashed')
        ax.plot([RK[0],RK[0]+RG1[0]],[RK[1],RK[1]+RG1[1]], color='red',ls='dashed')
        ax.plot([RK[0],RK[0]+RG2[0]],[RK[1],RK[1]+RG2[1]], color='red',ls='dashed')
        ax.plot([K[0],K[0]+g1[0]],[K[1],K[1]+g1[1]], color='black',ls='dashed')
        ax.plot([K[0],K[0]+g2[0]],[K[1],K[1]+g2[1]], color='black',ls='dashed')
    return K, RK, np.array([G1, G2]), np.array([RG1, RG2])
    

### get all k0+ig1+ig2 points around K and RK###
def get_bloch_states_basis(k0, k_cut, K, K_prim, RK, RK_prim, recip_latt_vec_moire):
    """
    For low-energy approximation, ks for unrotated and rotated layers should be around corresponds BZ cornors
    g1 = G1 - RG1
    g2 = G2 - RG2 
    g1 and g2 are equal to the reciprocal lattice vectors of the Moire pattern
    ki = k0 + m*g1 +n*g2
    ki's around K, K_prim, RK, RK_prim inside the circle with cutoff k_cut

    INPUTs:
        k0: k vector inside Moire BZ
        k_cut: only ki's inside 
        K, K_prim, RK, RK_prim: the K and K-prim points of the unrotted and rotted layers
        g1 and g2: 
    """
    g1, g2 = recip_latt_vec_moire
    def get_two_Tangent_lines(g, orig):
        """
        get two tangent lines of circle (orign: orig, radius: k_cut) parallel with vector g
        lines are described by ax+by+c=0
        """
        if g[0] == 0:
            a = 1
            b = 0
        else:
            a = g[1]/g[0]
            b = -1
        c0 = k_cut*np.sqrt(a**2+b**2)-(a*orig[0]+b*orig[1])
        c1 = -k_cut*np.sqrt(a**2+b**2)-(a*orig[0]+b*orig[1])
        return a, b, [c0, c1]
    
    def get_interaction_two_lines(a1,b1,c1, a2,b2,c2):
        """
        between a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0
        """
        x = (c2*b1-b2*c1)/(a1*b2-b1*a2)
        y = (a2*c1-a1*c2)/(a1*b2-b1*a2)
        return [x,y]

    def get_interactions(orig):
        """
        four interactions of the tangent lines of circle (orig and radius k_cut) parallel to g1 and g2
        """
        a1, b1, c1s = get_two_Tangent_lines(g1, orig)
        a2, b2, c2s = get_two_Tangent_lines(g2, orig)
        ps = [get_interaction_two_lines(a1,b1,c1s[i],a2,b2,c2s[j]) for i in [0,1] for j in [0,1]]
        fracs = cart2frac(np.array(ps)-k0, np.array([g1, g2]))
        return fracs

    def get_mn_range(orig):
        fracs = get_interactions(orig)
        m0 = np.int(np.min(fracs[:,0])-1)
        m1 = np.int(np.max(fracs[:,0])+1)
        n0 = np.int(np.min(fracs[:,1])-1)
        n1 = np.int(np.max(fracs[:,1])+1)
        return [m0, m1], [n0, n1]

    def get_mns_in_kcut(orig):
        ms,ns = get_mn_range(orig)
        fracs = np.array([[i,j] for i in range(ms[0], ms[1]+1) for j in range(ns[0],ns[1]+1)])
        ks = np.array([k0+i[0]*g1+i[1]*g2 for i in fracs])
        inds = np.where(np.linalg.norm(ks-orig, axis=1)<=k_cut)[0]
        return fracs[inds]
    return get_mns_in_kcut(K), get_mns_in_kcut(K_prim), get_mns_in_kcut(RK), get_mns_in_kcut(RK_prim)

#def get_interlayer_hopping_pairs(mns_K, mns_K_prim, mns_RK, mns_RK_prim, G1, G2, RG1, RG2):
#    """
#    hopping pairs with only t(|K|) selected
#     
#    About the k points of Bloch basis set: 
#    For unrotated layer: k0+i*g1+j*g2 for [i,j] in mns_K
#    For rotated layer: k0+i*g1+j*g2 for [i,j] in mns_RK
#    k2 - k1 = 0 or g1 or g2 when k1 is around K, t(q) ~ t(|K|)
#    """
#    ks_frac_K_00 = list(map(tuple, np.array(mns_K)))
#    ks_frac_K_01 = list(map(tuple, np.array(mns_K) + np.array([0,1])))
#    ks_frac_K_10 = list(map(tuple, np.array(mns_K) + np.array([1,0])))
#    ks_frac_RK_set = list(map(tuple, mns_RK))
#    intersect_00 = list(set(ks_frac_K_00) & set(ks_frac_RK_set))
#    intersect_01 = list(set(ks_frac_K_01) & set(ks_frac_RK_set))
#    intersect_10 = list(set(ks_frac_K_10) & set(ks_frac_RK_set))
#    inds_K_00 = [ks_frac_K_00.index(i) for i in intersect_00]
#    inds_RK_00 = [ks_frac_RK_set.index(i) for i in intersect_00]
#
#    inds_K_01 = [ks_frac_K_01.index(i) for i in intersect_01]
#    inds_RK_01 = [ks_frac_RK_set.index(i) for i in intersect_01]
#
#    inds_K_10 = [ks_frac_K_10.index(i) for i in intersect_10]
#    inds_RK_10 = [ks_frac_RK_set.index(i) for i in intersect_10]
#
#    ids_ks_K = np.concatenate((inds_K_00, inds_K_01, inds_K_10))
#    ids_ks_RK = np.concatenate((inds_RK_00, inds_RK_01, inds_RK_10))
#    return ids_ks_K, ids_ks_RK
    
### Hamiltonian matrix element between layers ###
def get_hop_func(h=3.35, g0=2.7, a0=1.42, g1=0.48, h0=3.35):
    def hop_func(r):
        dr = np.sqrt(r**2 + h**2)
        n = h/dr
        V_pppi = -g0 * np.exp(2.218 * (a0 - dr))
        V_ppsigma = g1 * np.exp(2.218 * (h0 - dr))
        hop = n**2*V_ppsigma + (1-n**2)*V_pppi
        return hop
    return hop_func

def hop_func_interlayer_FT(hop_func=get_hop_func(), R=5., dr=0.01, a=2.46):
    """
    the Fourier transform of interlayer hopping
    it can be transformed to be Hankel transform  
    tq = 1/s * int_0^infinity T(r)J0(kr)r dr
    """
    def hop_func_FT(qs):
        #qs = np.linalg.norm(qs, axis=2)
        shape = qs.shape
        qs = qs.reshape(1,-1)[0]
        ht = HankelTransform(nu=0, N=round(R/dr), h=dr)
        ft_qs = 2*np.pi * ht.transform(hop_func, qs, ret_err=False)/(a**2*np.sin(np.pi/3))
        return ft_qs.reshape(shape)
    return hop_func_FT

def get_Hamilt_interlayer_TB(ks_unrott,  mns_unrott, recip_latt_vec_unrott, tau_Xs_unrott, \
                                      ks_rott,    mns_rott,   recip_latt_vec_rott,   tau_Xs_rott):
    """
    k_unrott, k_rott: the k vecs in unrotated and rotated layers
    mn_unrott, mn_rott: the unit cell in unrotated and rotated reciprocal lattices 
    tau_Xs_unrott, tau_Xs_rott: the sublattice positions in unit cells
    coupling rule: k_unrott + G_unrott =  k_rott + G_rott
    order: right or left  right means basis order 1st:ks_unrott 2nd:ks_rott
                          left means basis order 1st:ks_rott 2nd:ks_unrott
    """
    G1, G2 = recip_latt_vec_unrott
    RG1, RG2 = recip_latt_vec_rott
    nk_unrott = len(ks_unrott)
    nk_rott = len(ks_rott)
    ms_unrott = mns_unrott[:,0]
    ns_unrott = mns_unrott[:,1] 
    ms_rott = mns_rott[:,0]
    ns_rott = mns_rott[:,1] 

    mmx, mmy = np.meshgrid(ms_rott, ms_unrott)
    nnx, nny = np.meshgrid(ns_rott, ns_unrott)
    out = np.zeros([nk_unrott, nk_rott, 2, 2])
    G_unrott = np.kron(mmy-mmx, G1).reshape(nk_unrott, nk_rott, 2) + np.kron(nny-nnx, G2).reshape(nk_unrott, nk_rott,2)
    G_rott = np.kron(mmy-mmx, RG1).reshape(nk_unrott, nk_rott, 2) + np.kron(nny-nnx, RG2).reshape(nk_unrott, nk_rott,2)
    normk = np.linalg.norm(np.transpose([ks_unrott]*nk_rott,axes=(1,0,2))  + G_unrott, axis=2)
    tk = hop_func_interlayer_FT()(normk) 
    M00 = np.exp(1j*np.sum(G_unrott*tau_Xs_unrott[0],axis=2)) * np.exp(-1j*np.sum(G_rott*tau_Xs_rott[0],axis=2))
    M01 = np.exp(1j*np.sum(G_unrott*tau_Xs_unrott[0],axis=2)) * np.exp(-1j*np.sum(G_rott*tau_Xs_rott[1],axis=2))
    M10 = np.exp(1j*np.sum(G_unrott*tau_Xs_unrott[1],axis=2)) * np.exp(-1j*np.sum(G_rott*tau_Xs_rott[0],axis=2))
    M11 = np.exp(1j*np.sum(G_unrott*tau_Xs_unrott[1],axis=2)) * np.exp(-1j*np.sum(G_rott*tau_Xs_rott[1],axis=2))
    out[:,:,0,0] = tk*M00
    out[:,:,0,1] = tk*M01
    out[:,:,1,0] = tk*M10
    out[:,:,1,1] = tk*M11
    return out.reshape(nk_unrott*2, nk_rott*2)

def get_Hamilt_interlayer_TB_old(k_unrott,  mn_unrott, recip_latt_vec_unrott, tau_Xs_unrott, \
                               k_rott,    mn_rott,   recip_latt_vec_rott,   tau_Xs_rott):
    """
    k_unrott, k_rott: the k vecs in unrotated and rotated layers
    mn_unrott, mn_rott: the unit cell in unrotated and rotated reciprocal lattices 
    tau_Xs_unrott, tau_Xs_rott: the sublattice positions in unit cells
    coupling rule: k_unrott + G_unrott =  k_rott + G_rott
    """
    G1, G2 = recip_latt_vec_unrott
    RG1, RG2 = recip_latt_vec_rott
    G_unrott = (mn_rott[0]-mn_unrott[0])*G1 + (mn_rott[1]-mn_unrott[1])*G2
    G_rott = (mn_rott[0]-mn_unrott[0])*RG1 + (mn_rott[1]-mn_unrott[1])*RG2
    
    normk = np.linalg.norm( k_unrott + G_unrott)
    tk = hop_func_interlayer_FT()(normk) 
    M00 = np.exp(1j*np.dot(G_unrott,tau_Xs_unrott[0])) * np.exp(-1j*np.dot(G_rott,tau_Xs_rott[0]))
    M01 = np.exp(1j*np.dot(G_unrott,tau_Xs_unrott[0])) * np.exp(-1j*np.dot(G_rott,tau_Xs_rott[1]))
    M10 = np.exp(1j*np.dot(G_unrott,tau_Xs_unrott[1])) * np.exp(-1j*np.dot(G_rott,tau_Xs_rott[0]))
    M11 = np.exp(1j*np.dot(G_unrott,tau_Xs_unrott[1])) * np.exp(-1j*np.dot(G_rott,tau_Xs_rott[1]))
    M = np.array([[M00, M01],[M10, M11]])
    return tk*M

### construct Hamiltonian
class ContinummModel:
    def __init__(self, a=2.46, h=3.35, origins=['atom','atom'], r_cut=2.0):
        """
        a: graphene lattice constant
        h: interlayer distance
        origins: origins for two layers
        r_cut: C-C distance cutoff for intra-layer hopping
        """
        self.a = a
        self.h = h
        self.origins = origins
        self.r_cut = r_cut

    def make_structure(self, theta):
        def get_sites_frac(origin):
            if origin == 'atom':
                sites_frac = [[0,0],[1/3., 1/3.]]
            elif origin == 'hole':
                sites_frac = [[1/3., 1/3.],[2/3., 2/3.]]
            return np.array(sites_frac)

        def get_latt_vec(theta):
            latt_vecs = self.a*np.array([[np.cos(np.pi/6), -1/2],\
                                         [np.cos(np.pi/6),  1/2]])
            return rotate_on_vec(theta, latt_vecs)
        latt_vec_unrott = get_latt_vec(0)
        latt_vec_rott = get_latt_vec(theta)
        sites_frac_unrott = get_sites_frac(self.origins[0])
        sites_frac_rott = get_sites_frac(self.origins[1])

        self.graphene_unrott = Graphene(self.h, latt_vec_unrott, sites_frac_unrott)
        self.graphene_unrott.add_hopping_pz(Rcut_intra=self.r_cut, g0=2.7, g1=0.48)
        self.graphene_rott = Graphene(self.h, latt_vec_rott, sites_frac_rott)
        self.graphene_rott.add_hopping_pz(Rcut_intra=self.r_cut, g0=2.7, g1=0.48)
             
        self.K, self.RK, self.recip_latt_vec_unrott, self.recip_latt_vec_rott = get_K_and_RK(latt_vec_unrott, latt_vec_rott)
        self.K_prim = -self.K
        self.RK_prim = -self.RK
        g1 = self.recip_latt_vec_unrott[0] - self.recip_latt_vec_rott[0]
        g2 = self.recip_latt_vec_unrott[1] - self.recip_latt_vec_rott[1]
        self.recip_latt_vec_moire = np.array([g1, g2])

    def _get_Bloch_basis(self, k0, k_cut):
        mns_K, mns_K_prim, mns_RK, mns_RK_prim = get_bloch_states_basis(k0, k_cut, self.K, self.K_prim,\
                                    self.RK, self.RK_prim, self.recip_latt_vec_moire)
        ks_K = frac2cart(mns_K, self.recip_latt_vec_moire)+ k0
        ks_K_prim = frac2cart(mns_K_prim, self.recip_latt_vec_moire)+ k0
        ks_RK = frac2cart(mns_RK, self.recip_latt_vec_moire)+ k0
        ks_RK_prim = frac2cart(mns_RK_prim, self.recip_latt_vec_moire)+ k0
        self.basis = {'K':ks_K, 'RK':ks_RK, 'K_prim':ks_K_prim, 'RK_prim':ks_RK_prim}
        self.mns_basis = {'K':mns_K, 'RK':mns_RK, 'K_prim':mns_K_prim, 'RK_prim':mns_RK_prim}

    def get_Hamiltonian_old(self, k0, k_cut):
        """
        k0: wave vector in Moire BZ
        k_cut: basis cutoff for kpoints around K, K_prim, RK, RK_prim
        r_cut: C-C distance cutoff for intra-layer hopping
        
        """
        tau_Xs_unrott = self.graphene_unrott.coords[:,0:2]
        tau_Xs_rott = self.graphene_rott.coords[:,0:2]
        recip_latt_vec_unrott = self.recip_latt_vec_unrott
        recip_latt_vec_rott = self.recip_latt_vec_rott
        
        self._get_Bloch_basis(k0, k_cut)
        nk_K = len(self.mns_basis['K'])
        nk_RK = len(self.mns_basis['RK'])
        nk_K_prim = len(self.mns_basis['K_prim'])
        nk_RK_prim = len(self.mns_basis['RK_prim'])
        ndim = (nk_K + nk_RK + nk_K_prim + nk_RK_prim)*2
        H = np.zeros((ndim, ndim), dtype=complex)
        ## basis order:  ks_K, ks_RK, ks_K_prim, ks_RK_prim
        ## intra-layer hopping
        def put_diag_mats_intra(k, graphene, ind, id_from):
            """
            ind: the order of the k 
            id0: the first ind 
            """
            H[id_from+2*ind:id_from+2*(ind+1), id_from+2*ind:id_from+2*(ind+1)] = graphene.hamilt_site_diff(k)
        [put_diag_mats_intra(self.basis['K'][i], self.graphene_unrott, i, 0) for i in range(nk_K)]
        [put_diag_mats_intra(self.basis['RK'][i], self.graphene_rott, i, 2*nk_K) for i in range(nk_RK)]
        [put_diag_mats_intra(self.basis['K_prim'][i], self.graphene_unrott, i, 2*nk_K+2*nk_RK) for i in range(nk_K_prim)]
        [put_diag_mats_intra(self.basis['RK_prim'][i], self.graphene_rott, i, 2*nk_K+2*nk_RK+2*nk_K_prim) for i in range(nk_RK_prim)]
        ## inter-layer hopping
        def put_diag_mats_inter(k_unrott, mn_unrott, k_rott, mn_rott, id0, id0_from, id1, id1_from):
            t = get_Hamilt_interlayer_TB(k_unrott, mn_unrott, recip_latt_vec_unrott, tau_Xs_unrott, \
                                           k_rott,   mn_rott,   recip_latt_vec_rott,   tau_Xs_rott)
            H[id0_from+2*id0:id0_from+2*(id0+1), id1_from+2*id1:id1_from+2*(id1+1)] = t
            H[id1_from+2*id1:id1_from+2*(id1+1), id0_from+2*id0:id0_from+2*(id0+1)] = np.transpose(t).conj()
        [put_diag_mats_inter(self.basis['K'][i], self.mns_basis['K'][i], self.basis['RK'][j], self.mns_basis['RK'][j], i, 0, j, 2*nk_K) \
               for i in range(nk_K) for j in range(nk_RK)]
        [put_diag_mats_inter(self.basis['K'][i], self.mns_basis['K'][i], self.basis['RK_prim'][j], self.mns_basis['RK_prim'][j],\
                i, 0, j, 2*nk_K+2*nk_RK+2*nk_K_prim) for i in range(nk_K) for j in range(nk_RK_prim)]
        [put_diag_mats_inter(self.basis['K_prim'][i], self.mns_basis['K_prim'][i], self.basis['RK'][j], self.mns_basis['RK'][j],\
                i, 2*nk_K+2*nk_RK, j, 2*nk_K) for i in range(nk_K_prim) for j in range(nk_RK)]
        [put_diag_mats_inter(self.basis['K_prim'][i], self.mns_basis['K_prim'][i], self.basis['RK_prim'][j], self.mns_basis['RK_prim'][j],\
                i, 2*nk_K+2*nk_RK, j, 2*nk_K+2*nk_RK+2*nk_K_prim) for i in range(nk_K_prim) for j in range(nk_RK_prim)]
        return H
    
    def get_Hamiltonian(self, k0, k_cut):
        """
        k0: wave vector in Moire BZ
        k_cut: basis cutoff for kpoints around K, K_prim, RK, RK_prim
        r_cut: C-C distance cutoff for intra-layer hopping
        
        """
        tau_Xs_unrott = self.graphene_unrott.coords[:,0:2]
        tau_Xs_rott = self.graphene_rott.coords[:,0:2]
        recip_latt_vec_unrott = self.recip_latt_vec_unrott
        recip_latt_vec_rott = self.recip_latt_vec_rott
        
        self._get_Bloch_basis(k0, k_cut)
        nk_K = len(self.mns_basis['K'])
        nk_RK = len(self.mns_basis['RK'])
        nk_K_prim = len(self.mns_basis['K_prim'])
        nk_RK_prim = len(self.mns_basis['RK_prim'])
        ndim = (nk_K + nk_RK + nk_K_prim + nk_RK_prim)*2
        H = np.zeros((ndim, ndim), dtype=complex)
        ## basis order:  ks_K, ks_RK, ks_K_prim, ks_RK_prim
        ## intra-layer hopping
        def put_diag_mats_intra(k, graphene, ind, id_from):
            """
            ind: the order of the k 
            id0: the first ind 
            """
            H[id_from+2*ind:id_from+2*(ind+1), id_from+2*ind:id_from+2*(ind+1)] = graphene.hamilt_site_diff(k)
        [put_diag_mats_intra(self.basis['K'][i], self.graphene_unrott, i, 0) for i in range(nk_K)]
        [put_diag_mats_intra(self.basis['RK'][i], self.graphene_rott, i, 2*nk_K) for i in range(nk_RK)]
        [put_diag_mats_intra(self.basis['K_prim'][i], self.graphene_unrott, i, 2*nk_K+2*nk_RK) for i in range(nk_K_prim)]
        [put_diag_mats_intra(self.basis['RK_prim'][i], self.graphene_rott, i, 2*nk_K+2*nk_RK+2*nk_K_prim) for i in range(nk_RK_prim)]
        ## inter-layer hopping
        ### K-->RK ###
        t_K_RK = get_Hamilt_interlayer_TB(self.basis['K'],  self.mns_basis['K'], recip_latt_vec_unrott, tau_Xs_unrott, \
                                           self.basis['RK'], self.mns_basis['RK'], recip_latt_vec_rott,   tau_Xs_rott)
        H[0:2*nk_K, 2*nk_K:2*(nk_K+nk_RK)] = t_K_RK
        H[2*nk_K:2*(nk_K+nk_RK), 0:2*nk_K] = np.transpose(t_K_RK).conj() 
        ### K-->RK' ###
        t_K_RK_prim = get_Hamilt_interlayer_TB(self.basis['K'],  self.mns_basis['K'], recip_latt_vec_unrott, tau_Xs_unrott, \
                                           self.basis['RK_prim'], self.mns_basis['RK_prim'], recip_latt_vec_rott,   tau_Xs_rott)
        H[0:2*nk_K, 2*(nk_K+nk_RK+nk_K_prim):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim)] = t_K_RK_prim
        H[2*(nk_K+nk_RK+nk_K_prim):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim), 0:2*nk_K] = np.transpose(t_K_RK_prim).conj() 
        ### K'-->RK ###
        t_K_prim_RK = get_Hamilt_interlayer_TB(self.basis['K_prim'],  self.mns_basis['K_prim'], recip_latt_vec_unrott, tau_Xs_unrott, \
                                           self.basis['RK'], self.mns_basis['RK'], recip_latt_vec_rott,   tau_Xs_rott)
        H[2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim), 2*nk_K:2*(nk_K+nk_RK)] = t_K_prim_RK
        H[2*nk_K:2*(nk_K+nk_RK), 2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim)] = np.transpose(t_K_prim_RK).conj() 
        ### K'-->RK' ###
        t_K_prim_RK_prim = get_Hamilt_interlayer_TB(self.basis['K_prim'],  self.mns_basis['K_prim'], recip_latt_vec_unrott, tau_Xs_unrott, \
                                           self.basis['RK_prim'], self.mns_basis['RK_prim'], recip_latt_vec_rott,   tau_Xs_rott)
        H[2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim), 2*(nk_K+nk_RK+nk_K_prim):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim)] = t_K_prim_RK_prim
        H[2*(nk_K+nk_RK+nk_K_prim):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim), 2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim)] = np.transpose(t_K_prim_RK_prim).conj() 
        return H

def special_ks(cm):
    recip_latt_vec = cm.recip_latt_vec_moire
    gamma = [0,0]
    M = recip_latt_vec[0]/2
    K = 1/3.*(2*recip_latt_vec[0] + recip_latt_vec[1])
    return [gamma, M, K, gamma]

        
def ks_path_sampling(ks, nk=20):
    nk_nodes = len(ks)
    k_region_norm = np.linalg.norm([ks[(i+1)]-ks[i] for i in range(nk_nodes-1)], axis=1)
    k_norm = np.sum(k_region_norm)
    n_pnts = [np.int(nk*k_region_norm[i]/k_norm) for i in range(nk_nodes-1)]
    kpts = []
    ids_node = []
    k = 0
    for i in range(nk_nodes-1):
        k0 = ks[i]
        k1 = ks[i+1]
        nki = n_pnts[i]
        dk = k_region_norm[i]/nki * ((k1-k0)/np.linalg.norm(k1-k0))
        ids_node.append(k)
        for j in range(nki):
            kpts.append(k0 + j*dk)
            k = k + 1
    kpts.append(ks[-1])
    ids_node.append(k)
    return kpts, ids_node

def plot_band(eig_f='EIGEN.npz', ylim=None, show=True):
    eig = np.load(eig_f, allow_pickle=True)
    kpts =eig['ks']
    dk = np.linalg.norm(kpts[1]-kpts[0])
    labels = eig['labels']

    nk = len(kpts)
    ks = np.linspace(0,nk*dk, nk)

    from matplotlib import pyplot as plt
    from mpl_toolkits import axisartist
    #import tBG
    #plt.rcParams.update(tBG.params)
    fig = plt.figure(1)
    ax = axisartist.Subplot(fig, 111)
    fig.add_subplot(ax)
    for i in range(nk):
        k = ks[i]
        enes = eig['vals'][i]
        ax.scatter([k]*len(enes), enes, color='black')
    #for i in k_info['inds']:
    #    ax.axvline(ks[i], linestyle='dashed', color='black', linewidth=0.5)
    ax.axhline(0.0, linestyle='dashed', color='black',linewidth=0.5)
    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
    #ax.set_xticks([ks[i] for i in k_info['inds']])
    #ax.set_xticklabels(labels)
    #ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    #ax.set_title('$\mathbf{\\rightarrow x}$')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(min(ks), max(ks))
    #fig.suptitle(title, x=0.5, y=1.01, fontsize=title_size)
    plt.tight_layout(w_pad=0.01, h_pad=0.01)
    if show:
        plt.show()

    
if __name__=='__main__':
    k_cut = 1.0
    cm = ContinummModel()
    cm.make_structure(4)
    kpath = special_ks(cm)
    kpts, ids_node = ks_path_sampling(kpath, nk=100)
    labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
    vals = []
    i=0
    for k in kpts:
        print(i)
        Hk = cm.get_Hamiltonian(k, k_cut)
        vals_k, vcs_k, info = zheev(Hk, 0)
        if info:
            raise ValueError('zheev failed')
        vals.append(vals_k)
        i = i + 1
    np.savez_compressed('EIGEN', vals=vals, ks=kpts, ids_node=ids_node, labels=labels)    
    
