from tBG.crystal.structures import CommensuStruct_pq
from tBG.crystal.brillouin_zones import BZHexagonal
from tBG.utils import cart2frac, rotate_on_vec, frac2cart
import numpy as np
from tBG.crystal.structures import Graphene
from hankel import HankelTransform
from scipy.linalg.lapack import zheev
import time
import math
from tBG.molecule.structures_new import coords_to_strlist

"""
repeat the small twisted angle part in PRB 93, 035452(2016)
"""

def get_basic_params(latt_vec_unrott, latt_vec_rott, latt_vec_moire, tau_Xs_unrott, tau_Xs_rott, plot=True):
    """
    return 
     (1) BZ cornors, reciprocal lattice vectors of unrott, rott and moire systems
     (2) coupling parameters:
         a, coupling lattice vectors [g1c, g2c]
         b, three most strongest coupling vectors gci (i=0,1,2) makes 
            k2 - k1 = gci
            gci = G1i - RG1i
            gci = mi*g1c + ni*g2c
            G1i = mi*b1 + ni*b2
            RG2i = mi*Rb1 + ni*Rb2
    """
    def get_K_Gs(latt_vec):
        bz = BZHexagonal(latt_vec)
        K = bz.special_points()['K'][0]
        b1 = bz.recip_latt_vec[0]
        b2 = bz.recip_latt_vec[1]
        return K, b1, b2
    K, b1, b2 = get_K_Gs(latt_vec_unrott)
    RK, Rb1, Rb2 = get_K_Gs(latt_vec_rott)
    Km, g1, g2 = get_K_Gs(latt_vec_moire)
    K_prim = -K
    RK_prim = -RK

    ### coupling lattice vectors [g1c, g2c], 
    ### k2's and k1's can coupling if k2 - k1 = m*g1c + n*g2c (m,n any integers)
    g1c = b1- Rb1
    g2c = b2 - Rb2

    def get_coupling_params(k1):
        mns = np.array([[i,j] for i in [-1,0,1] for j in [-1,0,1]])
        gcs = np.matmul(mns, [g1c, g2c])
        G1s = np.matmul(mns, [b1, b2])
        RG2s = np.matmul(mns, [Rb1, Rb2])
        norms = np.linalg.norm(k1+G1s, axis=1)
        ids = np.argsort(norms)[0:3]
        mns = mns[ids]
        gcs = gcs[ids]
        G1s = G1s[ids]
        RG2s = RG2s[ids]
        Ms = []
        for i in range(len(ids)):
            G1 = G1s[i]; RG2 = RG2s[i]
            M00 = np.exp(1j*np.dot(G1, tau_Xs_unrott[0]))*np.exp(-1j*np.dot(RG2,tau_Xs_rott[0]))
            M01 = np.exp(1j*np.dot(G1, tau_Xs_unrott[0]))*np.exp(-1j*np.dot(RG2,tau_Xs_rott[1]))
            M10 = np.exp(1j*np.dot(G1, tau_Xs_unrott[1]))*np.exp(-1j*np.dot(RG2,tau_Xs_rott[0]))
            M11 = np.exp(1j*np.dot(G1, tau_Xs_unrott[1]))*np.exp(-1j*np.dot(RG2,tau_Xs_rott[1]))
            Ms.append([[M00, M01],[M10, M11]])
        return {'mns':mns, 'gcs':gcs, 'G1s':G1s, 'RG2s':RG2s, 'Ms':np.array(Ms)}
    
    spec_kpts = {'K':K, 'RK':RK, 'K_prim':K_prim, 'RK_prim':RK_prim}        
    recip_latt_vecs = {'unrott': np.array([b1,b2]), 'rott':np.array([Rb1, Rb2]), 'moire':np.array([g1,g2])}
    coupling_params = {'K':get_coupling_params(K), 'K_prim':get_coupling_params(K_prim), 'coupling_latt_vec':np.array([g1c,g2c])}

    if plot: 
        from matplotlib import pyplot as plt
        plt.plot([0, b1[0]], [0,b1[1]], label='b1')
        plt.plot([0, b2[0]], [0,b2[1]], label='b2')
        plt.plot([0, Rb1[0]], [0,Rb1[1]], label='Rb1')
        plt.plot([0, Rb2[0]], [0,Rb2[1]], label='Rb2')
        plt.plot([0, g1[0]], [0,g1[1]], label='g1')
        plt.plot([0, g2[0]], [0,g2[1]], label='g2')
        plt.legend()
        plt.axis('equal')
        plt.show()
    return spec_kpts, recip_latt_vecs, coupling_params

def get_k_basis_unrott(k_cut, recip_latt_vec_moire):
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
    def get_two_Tangent_lines(g):
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
        c0 = k_cut*np.sqrt(a**2+b**2)
        c1 = -k_cut*np.sqrt(a**2+b**2)
        return a, b, [c0, c1]
    
    def get_interaction_two_lines(a1,b1,c1, a2,b2,c2):
        """
        between a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0
        """
        x = (c2*b1-b2*c1)/(a1*b2-b1*a2)
        y = (a2*c1-a1*c2)/(a1*b2-b1*a2)
        return [x,y]

    def get_interactions():
        """
        four interactions of the tangent lines of circle (orig and radius k_cut) parallel to g1 and g2
        """
        a1, b1, c1s = get_two_Tangent_lines(g1)
        a2, b2, c2s = get_two_Tangent_lines(g2)
        ps = [get_interaction_two_lines(a1,b1,c1s[i],a2,b2,c2s[j]) for i in [0,1] for j in [0,1]]
        fracs = cart2frac(np.array(ps), np.array([g1, g2]))
        return fracs

    def get_mn_range():
        fracs = get_interactions()
        m0 = np.int(np.min(fracs[:,0])-1)
        m1 = np.int(np.max(fracs[:,0])+1)
        n0 = np.int(np.min(fracs[:,1])-1)
        n1 = np.int(np.max(fracs[:,1])+1)
        return [m0, m1], [n0, n1]

    def get_ks_in_kcut():
        ms,ns = get_mn_range()
        fracs = np.array([[i,j] for i in range(ms[0], ms[1]+1) for j in range(ns[0],ns[1]+1)])
        ks = np.array([i[0]*g1+i[1]*g2 for i in fracs])
        inds = np.where(np.linalg.norm(ks, axis=1)<=k_cut)[0]
        return ks[inds]

    dks_unrott = get_ks_in_kcut()
    return dks_unrott

def get_k_basis_rott(ks_unrott, gcs):
    prec = int(np.log10(4/np.linalg.norm(gcs[0]))) + 2
    ks_rott = []
    for i in range(len(gcs)):
        ks_rott.append(ks_unrott+gcs[i])
    ks_rott = np.concatenate(ks_rott, axis=0)
    ks_rott_str = coords_to_strlist(ks_rott, prec)
    inds = np.unique(ks_rott_str, return_index=True)[1]
    return ks_rott[inds]

def get_coupling_pairs(dks_unrott, dks_rott, gcs):
    prec = int(np.log10(4/np.linalg.norm(gcs[0]))) + 1
    n0 = len(dks_unrott)
    n1 = len(dks_rott)
    dks_unrott_str = coords_to_strlist(dks_unrott, prec)
    dks_rott_str = coords_to_strlist(dks_rott, prec)
    loc = dict(zip(dks_rott_str, range(n0,n0+n1)))
    pairs = []
    for i in range(len(gcs)):
        pairs_i = []
        dks_unrott_plus_gci = dks_unrott + gcs[i]
        dks_unrott_plus_gci_str = coords_to_strlist(dks_unrott_plus_gci, prec)
        for j in range(len(dks_unrott_plus_gci_str)):
            pairs_i.append([j,loc[dks_unrott_plus_gci_str[j]]])
        pairs.append(pairs_i)
    return pairs

def Hamiltonian_interlayer(nk_K, nk_RK, pairs, Ms, t=-0.11):
    ndim = 2*(nk_K+nk_RK)
    H = np.zeros([ndim, ndim], dtype=complex)
    for i in range(len(Ms)):
        M = Ms[i]
        pair = pairs[i]
        for p,q in pair:
            H[2*p:2*(p+1), 2*q:2*(q+1)] = t*M
            H[2*q:2*(q+1), 2*p:2*(p+1)] = np.transpose(t*M).conj()
    return H

############ t(q) interlayer coupling strength in reciprocal space #####################################
##### a fixed number -0.11 is used in this method, so this part is not used in the current version
##### t(|K|) = 0.11 eV, where |K| is the distance from origin to BZ cornors of the unrotated and rotated layers
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
        ht = HankelTransform(nu=0, N=round(R/dr), h=dr)
        ft_qs = 2*np.pi * ht.transform(hop_func, qs, ret_err=False)/(a**2*np.sin(np.pi/3))
        return ft_qs
    return hop_func_FT
################################################################################################


class ContinummModel:
    def __init__(self, a=2.46, h=3.35, rotate_cent='atom', r_cut=2.0):
        """
        a: graphene lattice constant
        h: interlayer distance
        origins: origins for two layers
        r_cut: C-C distance cutoff for intra-layer hopping
        k_cut: k basis size
        """
        self.a = a
        self.h = h
        self.rotate_cent = rotate_cent
        self.r_cut = r_cut

    def make_structure(self, p=1, q=5):

        cs = CommensuStruct_pq(rotate_cent=self.rotate_cent)
        cs.make_structure(p,q)

        latt_vec_unrott = cs.layer_latt_vecs[0]
        latt_vec_rott = cs.layer_latt_vecs[1]
        latt_vec_moire = cs.latt_vec[0:2][0:2]
        self.latt_vecs = {'unrott':latt_vec_unrott, 'rott':latt_vec_rott,'moire':latt_vec_moire}
        theta = np.arccos((3*q**2-p**2)/(3*q**2+p**2))
        print('theta:%f'%(theta*180/np.pi))
        if self.rotate_cent=='atom':
            sites_frac = [[0,0],[1/3,1/3]]
        elif self.rotate_cent=='hole':
            sites_frac = [[1/3.,1/3.],[2/3,2/3]]

        self.graphene_unrott = Graphene(self.h, latt_vec_unrott, sites_frac)
        self.graphene_unrott.add_hopping_pz(Rcut_intra=self.r_cut, g0=2.7, g1=0.48)
        self.graphene_rott = Graphene(self.h, latt_vec_rott, sites_frac)
        self.graphene_rott.add_hopping_pz(Rcut_intra=self.r_cut, g0=2.7, g1=0.48)

        tau_Xs_unrott = self.graphene_unrott.coords[:,0:2]
        tau_Xs_rott = self.graphene_rott.coords[:,0:2]

        self.spec_kpts, self.recip_latt_vecs, self.coupling_params = \
                  get_basic_params(latt_vec_unrott, latt_vec_rott, latt_vec_moire, tau_Xs_unrott, tau_Xs_rott)
        

    def _get_Bloch_basis(self, k_cut, plot=True):
        recip_latt_vec_moire = self.recip_latt_vecs['moire']
        dks_unrott = get_k_basis_unrott(k_cut, recip_latt_vec_moire)

        gcs_K = self.coupling_params['K']['gcs']
        gcs_K_prim = self.coupling_params['K_prim']['gcs']
        dks_rott_K = get_k_basis_rott(dks_unrott, gcs_K)
        dks_rott_K_prim = get_k_basis_rott(dks_unrott, gcs_K_prim)
        self.basis = {'unrott':dks_unrott, 'rott_K':dks_rott_K, 'rott_K_prim':dks_rott_K_prim}
        self.coupling_pairs = {'K':get_coupling_pairs(dks_unrott, dks_rott_K, gcs_K),\
                               'K_prim':get_coupling_pairs(dks_unrott, dks_rott_K_prim, gcs_K_prim)}
         
        if plot:
            K = self.spec_kpts['K']
            RK = self.spec_kpts['RK']
            K_prim = self.spec_kpts['K_prim']
            Ks = np.array([rotate_on_vec(i*60, K) for i in range(7)])
            RKs = np.array([rotate_on_vec(i*60, RK) for i in range(7)])
            from matplotlib import pyplot as plt
            plt.plot(Ks[:,0], Ks[:,1])
            plt.plot(RKs[:,0], RKs[:,1])
            Ks = dks_unrott + K
            RKs = dks_rott_K + K
            plt.scatter(Ks[:,0],Ks[:,1], s=150, label='K', marker='o')
            plt.scatter(RKs[:,0],RKs[:,1], label='RK',marker='s')
            Ks_prim = dks_unrott + K_prim
            RKs_prim = dks_rott_K_prim + K_prim
            plt.scatter(Ks_prim[:,0],Ks_prim[:,1], s=150, label='K\'', marker='o')
            plt.scatter(RKs_prim[:,0],RKs_prim[:,1], label='RK\'',marker='s')
            plt.legend()
            plt.axis('equal')
            plt.show()

    
    def get_Hamiltonian(self, k0, K_vally=True, Kprim_Vally=True):
        """
        k0: k vector around K_vally, -k0 is used for Kprim vally
        K_vally, and Kprim_vally, whether K and Kprim valleyes are involved
        """
        if K_vally==False and Kprim_vally==False:
            raise ValueError('At least one valley is involved!')

        dks_unrott = self.basis['unrott']
        dks_rott_K = self.basis['rott_K']
        dks_rott_K_prim = self.basis['rott_K_prim']

        ###
        nk_K = len(dks_unrott)
        nk_RK = len(dks_rott_K)
        pairs_K = self.coupling_pairs['K']
        Ms_K = cm.coupling_params['K']['Ms']    
        ks_K = dks_unrott + k0
        ks_RK = dks_rott_K + k0 

        ###
        nk_K_prim = len(dks_unrott)
        nk_RK_prim = len(dks_rott_K_prim)
        ndim = 2*(nk_K+nk_K_prim+nk_RK+nk_RK_prim)
        pairs_K_prim = self.coupling_pairs['K_prim']
        Ms_K_prim = cm.coupling_params['K_prim']['Ms']    
        ks_K_prim = dks_unrott - k0
        ks_RK_prim = dks_rott_K_prim - k0

        H = np.zeros([ndim, ndim], dtype=complex)
        ### interlayer ###

        H[0:2*(nk_K+nk_RK), 0:2*(nk_K+nk_RK)] = Hamiltonian_interlayer(nk_K, nk_RK, pairs_K, Ms_K, t=0.11)
        
        H_K_prim = Hamiltonian_interlayer(nk_K_prim, nk_RK_prim, pairs_K_prim, Ms_K_prim, t=0.11)
        H[2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim), 2*(nk_K+nk_RK):2*(nk_K+nk_RK+nk_K_prim+nk_RK_prim)] = H_K_prim

        def put_diag_mats_intra(k, graphene, ind, id_from):
            """
            ind: the order of the k 
            id0: the first ind 
            """
            H[id_from+2*ind:id_from+2*(ind+1), id_from+2*ind:id_from+2*(ind+1)] = graphene.hamilt_site_diff(k)
        
        [put_diag_mats_intra(ks_K[i], self.graphene_unrott, i, 0) for i in range(nk_K)]
        [put_diag_mats_intra(ks_RK[i], self.graphene_rott, i, 2*nk_K) for i in range(nk_RK)]
        [put_diag_mats_intra(ks_K_prim[i], self.graphene_unrott, i, 2*(nk_K+nk_RK)) for i in range(nk_K_prim)]
        [put_diag_mats_intra(ks_RK_prim[i], self.graphene_rott, i, 2*(nk_K+nk_RK+nk_K_prim)) for i in range(nk_RK_prim)]
        return H

def special_ks(cm):
    K = cm.spec_kpts['K']
    RK = cm.spec_kpts['RK']
    M = (K+RK)/2
    dm = M/np.linalg.norm(M)
    dk = RK - K
    A = K
    B = RK
    C = RK + dk
    
    bz = BZHexagonal(cm.latt_vecs['moire'])
    Mb = bz.special_points()['M'][0]
    OM = np.linalg.norm(Mb)
    
    D = M + dm*OM
    return [A, B, C, D, A]

        
def ks_path_sampling(ks, nk=50):
    nk_nodes = len(ks)
    k_region_norm = np.linalg.norm([ks[i+1]-ks[i] for i in range(nk_nodes-1)], axis=1)
    k_norm = np.sum(k_region_norm)
    print(k_region_norm, k_norm)
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

def plot_band(eig_f='EIGEN.npz', show=True, ylim=[-2,2]):
    eig = np.load(eig_f, allow_pickle=True)
    kpts =eig['ks']
    dk = np.linalg.norm(kpts[1]-kpts[0])
    labels = eig['labels']
    ids_nodes = eig['ids_node']

    nk = len(kpts)
    ks = np.linspace(0,nk*dk, nk)

    from matplotlib import pyplot as plt
    from mpl_toolkits import axisartist
    fig = plt.figure(1)
    ax = axisartist.Subplot(fig, 111)
    fig.add_subplot(ax)
    for i in range(nk):
        k = ks[i]
        enes = eig['vals'][i]
        ax.scatter([k]*len(enes), enes, color='black')
    for i in ids_nodes:
        ax.axvline(ks[i], linestyle='dashed', color='black', linewidth=0.5)
    ax.axhline(0.0, linestyle='dashed', color='black',linewidth=0.5)
    ax.set_ylabel('$\mathbf{Energy-E_f}$ (eV)')
    ax.set_xticks([ks[i] for i in ids_nodes])
    ax.set_xticklabels(labels)
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
    k_cut = 1
    cm = ContinummModel()
    cm.make_structure(p=1, q=61)
    cm._get_Bloch_basis(0.2)
    kpath = special_ks(cm)
    kpts, ids_node = ks_path_sampling(kpath, nk=100)
    labels = ['A', 'B', 'C', 'D', 'A']
    vals = []
    i=0
    for k in kpts:
        print(i, k)
        Hk = cm.get_Hamiltonian(k, k_cut)
        vals_k, vcs_k, info = zheev(Hk, 0)
        if info:
            raise ValueError('zheev failed')
        vals.append(vals_k)
        i = i + 1
    np.savez_compressed('EIGEN', vals=vals, ks=kpts, ids_node=ids_node, labels=labels)    
    
