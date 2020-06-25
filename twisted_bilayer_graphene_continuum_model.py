from tBG.periodic_structures import CommensuStruct
from tBG.brillouin_zones import BZHexagonal
from tBG.utils import cart2frac
import numpy as np

"""
repeat the small twisted angle part in PRB 93, 035452(2016)
"""

### get K and RK ###
def get_K_and_RK(latt_vec_unrott, latt_vec_rott, ax=None):
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
    return K, RK, g1, g2
    

### get all k0+ig1+ig2 points around K and RK###
def get_bloch_states_basis(k0, k_cut, K, RK, g1, g2):
    """
    for low-energy approximation, ks for unrotated and rotated layers should be around corresponds BZ cornors
    k1 = k0 + i*g1 +j*g2
    k2 = k0 + i*g1 +j*g2
    k1 around K, k2 around RK (inside the circle with origin K or RK and radius k_cut)
    """
    K = np.array(K)
    RK = np.array(RK)
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
    return get_mns_in_kcut(K), get_mns_in_kcut(RK)

def get_interlayer_hopping_pairs(mns_K, mns_RK):
    """
    selection rules for interlayer hopping
     
    About the k points of Bloch basis set: 
    For unrotated layer: k0+i*g1+j*g2 for [i,j] in mns_K
    For rotated layer: k0+i*g1+j*g2 for [i,j] in mns_RK
    k2 - k1 = 0 or g1 or g2 when k1 is around K, t(q) ~ t(|K|)
    """
    ks_frac_K_00 = list(map(tuple, np.array(mns_K)))
    ks_frac_K_01 = list(map(tuple, np.array(mns_K) + np.array([0,1])))
    ks_frac_K_10 = list(map(tuple, np.array(mns_K) + np.array([1,0])))
    ks_frac_RK_set = list(map(tuple, mns_RK))
    intersect_00 = list(set(ks_frac_K_00) & set(ks_frac_RK_set))
    intersect_01 = list(set(ks_frac_K_01) & set(ks_frac_RK_set))
    intersect_10 = list(set(ks_frac_K_10) & set(ks_frac_RK_set))
    inds_K_00 = [ks_frac_K_00.index(i) for i in intersect_00]
    inds_RK_00 = [ks_frac_RK_set.index(i) for i in intersect_00]

    inds_K_01 = [ks_frac_K_01.index(i) for i in intersect_01]
    inds_RK_01 = [ks_frac_RK_set.index(i) for i in intersect_01]

    inds_K_10 = [ks_frac_K_10.index(i) for i in intersect_10]
    inds_RK_10 = [ks_frac_RK_set.index(i) for i in intersect_10]

    ids_ks_K = np.concatenate((inds_K_00, inds_K_01, inds_K_10))
    ids_ks_RK = np.concatenate((inds_RK_00, inds_RK_01, inds_RK_10))
    return ids_ks_K, ids_ks_RK
    

### Hamiltonian matrix element within onelayer ###
def get_Hamilt_inlayer(k):
    return

### Hamiltonian matrix element between layers ###
def get_Hamilt_between_layers(k1,k2):
    """
    k1: the k vec in unrotated layer
    k2: the k vec in rotated layer
    k1 and k2 are relative to K and RK, respectively
    """
    return

### construct Hamiltonian
class ContinummModel:
    def __init__(self, a=2.46, h=3.35, rotate_cent='atom'):
        self.a = a
        self.h = h
        self.rotate_cent = rotate_cent

    def make_structure(self, m,n):
        cs = CommensuStruct(self.a, self.h, rotate_cent = self.rotate_cent)
        cs.make_structure(m, n)
        self.K, self.RK, self.g1, self.g2 = get_K_and_RK(cs.latt_vec_bott, cs.latt_vec_top)
        self.twist_angle = cs.twist_angle

    def _get_Bloch_basis(self, k0, k_cut):
        mns_unrott, mns_rott = get_bloch_states_basis(k0, k_cut, self.K, self.RK, self.g1, self.g2)
        Bloch_basis_unrott = np.array([k0+i[0]*self.g1+i[1]*self.g2 for i in mns_unrott])
        Bloch_basis_rott = np.array([k0+i[0]*self.g1+i[1]*self.g2 for i in mns_rott])
        inter_hopping_pairs = get_interlayer_hopping_pairs(mns_unrott, mns_rott)
        return Bloch_basis_unrott, Bloch_basis_rott, inter_hopping_pairs
    
    def get_Hamiltoinan(self, k0, k_cut):
        k_basis_unrott, k_basis_rott, inter_pairs = self._get_Bloch_basis(k0, k_cut)
        nk_unrott = len(basis_unrott)
        nk_rott = len(basis_rott)
        ndim = (nk_unrott + un_rott)*2
        H = np.zeros((ndim, ndim), dtype=complex)
    
if __name__=='__main__':
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.axis('equal')
    cm = ContinummModel()
    cm.make_structure(10,11)
    K,RK,g1,g2,G = get_K_and_RK(cs.latt_vec_bott, cs.latt_vec_top, ax=ax)
    k_cut = 1/4*np.linalg.norm(G)/np.sqrt(3)
    k0 = np.array([0,0])
    mns_K, mns_RK = get_bloch_states_basis(k0, k_cut, K, RK, g1, g2)
    ks_K = np.array([k0+i[0]*g1+i[1]*g2 for i in mns_K])
    ks_RK = np.array([k0+i[0]*g1+i[1]*g2 for i in mns_RK])
    ax.scatter(ks_K[:,0], ks_K[:,1], color='black', marker='s')
    ax.scatter(ks_RK[:,0], ks_RK[:,1], color='red')
    pairs = get_interlayer_hopping_pairs(mns_K, mns_RK)
    for i in range(len(pairs[0])):
        ax.plot([ks_K[pairs[0][i]][0], ks_RK[pairs[1][i]][0]], \
                [ks_K[pairs[0][i]][1], ks_RK[pairs[1][i]][1]], color='green')
    plt.show()
