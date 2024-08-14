import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def optimal_network(Z, data):
    Z = sorted(Z)
    _, kVar = np.shape(data)
    DAG = np.zeros((kVar, kVar))
    data_array = np.array(data, dtype=np.int_)

    
    while kVar > 0:
        kVar -= 1
        if kVar not in Z:
           data_array = np.delete(data_array, kVar, axis=1)

    
    z_dict = hc(data_array, metric="BIC")
    #print("z_dict= ",z_dict)
    c_dict = dict()
    for key, value in z_dict.items():        
        if value == []:
            c_dict.setdefault(Z[key], [])
        else:          
            c_list = []
            for i in value:
                c_list.append(Z[i])
                DAG[Z[key], Z[i]] = 1
            c_dict.setdefault(Z[key], c_list)
    return DAG



def S2TMB(data, target):
    print("S2TMB Input data shape:", data.shape)
    print("S2TMB Target:", target)

    _, kVar = np.shape(data)
    pc_t = []
    o_set = [i for i in range(kVar) if i != target]
    #print("KVAR=",kVar,o_set)
    for x in o_set:
        Z = set([target, x]).union(pc_t)
        #print("Z set:", Z)
        DAG = optimal_network(Z, data)
        #print("DAG output: ",DAG)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        #print("pc_t after optimal_network:", pc_t)

    spouses_t = []
    varis_set = [i for i in range(kVar) if i != target and i not in pc_t]
    for x in varis_set:
        Z = set([target, x]).union(set(pc_t)).union(set(spouses_t))
        #print("Z set for spouses:", Z)
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        spouses_t = [i for i in range(kVar) for j in range(kVar) if i != target and DAG[target, j] == 1 and DAG[i, j] == 1]

    MB = list(set(pc_t).union(set(spouses_t)))
    #print("MB (Markov Blanket):", MB)
    return pc_t, MB


def hc(data, metric='AIC', max_iter=10, debug=True, restriction=None):

    nrow = data.shape[0]
    ncol = data.shape[1]
    
    names = range(ncol)

    # INITIALIZE NETWORK W/ NO EDGES
    # maintain children and parents dict for fast lookups
    c_dict = dict([(n, []) for n in names])
    p_dict = dict([(n, []) for n in names])
    
    # COMPUTE INITIAL LIKELIHOOD SCORE	
    value_dict = dict([(n, np.unique(data[:, i])) for i,n in enumerate(names)])
    bn = BayesNet(c_dict)
    mle_estimator(bn, data)
    max_score = info_score(bn, nrow, metric)
    #print("Initial max score:", max_score)

    # CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
    #ED = EmpiricalDistribution(data,names)

    

    _iter = 0
    improvement = True

    while improvement:
        improvement = False
        max_delta = 0

        if debug:
            print('ITERATION: ', _iter)

        ### TEST ARC ADDITIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v not in c_dict[u] and u != v and not would_cause_cycle(c_dict, u, v):
                    # FOR MMHC ALGORITHM -> Edge Restrictions
                    if restriction is None or (u, v) in restriction:
                        # SCORE FOR 'V' -> gaining a parent
                        old_cols = (v,) + tuple(p_dict[v])  # without 'u' as parent
                        mi_old = mutual_information(data[:, old_cols])
                        new_cols = old_cols + (u,) # with'u' as parent
                        mi_new = mutual_information(data[:, new_cols])
                        delta_score = nrow * (mi_old - mi_new)
                        

                        if delta_score > max_delta:
                            
                            if debug:
                                print('Improved Arc Addition: ' , (u,v))
                                print('Delta Score: ' , delta_score)
                            max_delta = delta_score
                            max_operation = 'Addition'
                            max_arc = (u,v)

        ### TEST ARC DELETIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u]:
                    # SCORE FOR 'V' -> losing a parent
                    old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = tuple([i for i in old_cols if i != u]) # without 'u' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta_score = nrow * (mi_old - mi_new)

                    if delta_score > max_delta:
                        if debug:
                            print('Improved Arc Deletion: ' , (u,v))
                            print('Delta Score: ' , delta_score)
                        max_delta = delta_score
                        max_operation = 'Deletion'
                        max_arc = (u,v)

        ### TEST ARC REVERSALS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u] and not would_cause_cycle(c_dict,v,u, reverse=True):
                    # SCORE FOR 'U' -> gaining 'v' as parent
                    old_cols = (u,) + tuple(p_dict[v]) # without 'v' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = old_cols + (v,) # with 'v' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta1 = nrow * (mi_old - mi_new)
                    # SCORE FOR 'V' -> losing 'u' as parent
                    old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = tuple([u for i in old_cols if i != u]) # without 'u' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta2 = nrow * (mi_old - mi_new)
                    # COMBINED DELTA-SCORES
                    delta_score = delta1 + delta2

                    if delta_score > max_delta:
                        if debug:
                            print('Improved Arc Reversal: ' , (u,v))
                            print('Delta Score: ' , delta_score)
                        max_delta = delta_score
                        max_operation = 'Reversal'
                        max_arc = (u,v)


        ### DETERMINE IF/WHERE IMPROVEMENT WAS MADE ###
        if max_delta > 0:
            improvement = True
            u,v = max_arc
            if max_operation == 'Addition':
                if debug:
                    print('ADDING: ' , max_arc , '\n')
                c_dict[u].append(v)
                p_dict[v].append(u)
            elif max_operation == 'Deletion':
                if debug:
                    print('DELETING: ' , max_arc , '\n')
                c_dict[u].remove(v)
                p_dict[v].remove(u)
            elif max_operation == 'Reversal':
                if debug:
                    print('REVERSING: ' , max_arc, '\n')
                    c_dict[u].remove(v)
                    p_dict[v].remove(u)
                    c_dict[v].append(u)
                    p_dict[u].append(v)
        else:
            if debug:
                print('No Improvement on Iter: ' , _iter)

        ### TEST FOR MAX ITERATION ###
        _iter += 1
        if _iter > max_iter:
            if debug:
                print('Max Iteration Reached')
            break
    
    #print("Final c_dict:", c_dict)
    return c_dict



def mutual_information(data, conditional=False):
    try:
        bins = np.amax(data, axis=0) + 1
        #print(f"Bins: {bins}")
        if len(bins) == 1:
            hist, _ = np.histogramdd(data, bins=(bins))
            Px = hist / hist.sum()
            MI = -1 * np.sum(Px * np.log(Px + 1e-7))  # Adding a small constant to avoid log(0)
            #print(f"MI: {MI}")
            return round(MI, 4)
        else:
            hist, _ = np.histogramdd(data, bins=bins)
            Px = hist / hist.sum()
            Px = Px.flatten()
            Px = Px[Px > 0]
            Hx = -np.sum(Px * np.log(Px + 1e-7))  # Adding a small constant to avoid log(0)
            MI = 0.0
            if conditional:
                for i in range(len(data.T) - 1):
                    Px_given = hist.sum(axis=i + 1) / hist.sum()
                    Px_given = Px_given.flatten()
                    Px_given = Px_given[Px_given > 0]
                    Hx_given = -np.sum(Px_given * np.log(Px_given + 1e-7))  # Adding a small constant to avoid log(0)
                    MI += Hx - Hx_given
            else:
                MI = Hx
            #print(f"MI: {MI}")
            return round(MI, 4)
    except Exception as e:
        print(f"Error calculating mutual information: {e}")
        return None






def topsort(edge_dict, root=None):
    """
    List of nodes in topological sort order from edge dict
    where key = rv and value = list of rv's children
    """
    queue = []
    if root is not None:
        queue = [root]
    else:
        for rv in edge_dict.keys():
            prior=True
            for p in edge_dict.keys():
                if rv in edge_dict[p]:
                    prior=False
            if prior==True:
                queue.append(rv)
    
    visited = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            for nbr in edge_dict[vertex]:
                queue.append(nbr)
            #queue.extend(edge_dict[vertex]) # add all vertex's children
    return visited


class BayesNet(object):
    """
    Overarching class for Bayesian Networks

    """

    def __init__(self, E=None, value_dict=None, file=None):
        """
        Initialize the BayesNet class.

        Arguments
        ---------
        *V* : a list of strings - vertices in topsort order
        *E* : a dict, where key = vertex, val = list of its children
        *F* : a dict, 
            where key = rv, 
            val = another dict with
                keys = 
                    'parents', 
                    'values', 
                    'cpt'

        *V* : a dict        

        Notes
        -----
        
        """
        if file is not None:
            import SSD.MBs.pyBN.io.read as ior
            bn = ior.read_bn(file)
            self.V = bn.V
            self.E = bn.E
            self.F = bn.F        
        else:
            if E is not None:
                #assert (value_dict is not None), 'Must set values if E is set.'
                self.set_structure(E, value_dict)
            else:
                self.V = []
                self.E = {}
                self.F = {}

    def __eq__(self, y):
        """
        Tests whether two Bayesian Networks are
        equivalent - i.e. they contain the same
        node/edge structure, and equality of
        conditional probabilities.
        """
        return are_class_equivalent(self, y)

    def __hash__(self):
        """
        Allows BayesNet objects to be used
        as keys in a dictionary (i.e. hashable)
        """
        return hash((str(self.V),str(self.E)))

    def copy(self):
        V = deepcopy(self.V)
        E = deepcopy(self.E)
        F = {}
        for v in V:
            F[v] = {}
            F[v]['cpt'] = deepcopy(self.F[v]['cpt'])
            F[v]['parents'] = deepcopy(self.F[v]['parents'])
            F[v]['values'] = deepcopy(self.F[v]['values'])
        bn = BayesNet()
        bn.V = V
        bn.E = E
        bn.F = F

        return bn

    def add_node(self, rv, cpt=[], parents=[], values=[]):
        self.V.append(rv)
        self.F[rv] = {'cpt':cpt,'parents':parents,'values':values}

    def add_edge(self, u, v):
        if not self.has_node(u):
            self.add_node(u)
        if not self.has_node(v):
            self.add_node(v)
        if self.has_edge(u,v):
            print('Edge already exists')
        else:
            self.E[u].append(v)
            self.F[v]['parents'].append(u)
        #self.V = topsort(self.E)
        # HOW DO I RECALCULATE CPT?


    def remove_edge(self, u, v):
        self.E[u].remove(v)
        self.F[v]['parents'].remove(u)

    def reverse_arc(self, u, v):
        if self.has_edge(u,v):
            self.E[u].remove(v)
            self.E[v].append(u)

    def set_data(self, rv, data):
        assert (isinstance(data, dict)), 'data must be dictionary'
        self.F[rv] = data

    def set_cpt(self, rv, cpt):
        self.F[rv]['cpt'] = cpt

    def set_parents(self, rv, parents):
        self.F[rv]['parents'] = parents

    def set_values(self, rv, values):
        self.F[rv]['values'] = values

    def nodes(self):
        for v in self.V:
            yield v

    def node_idx(self, rv):
        try:
            return self.V.index(rv)
        except ValueError:
            return -1

    def has_node(self, rv):
        return rv in self.V

    def has_edge(self, u, v):
        return v in self.E[u]

    def edges(self):
        for u in self.nodes():
            for v in self.E[u]:
                  (u,v)
    def num_edges(self):
        num = 0
        for u in self.nodes():
            num += len(self.E[u])
        return num

    def num_params(self):
        num = 0
        for u in self.nodes():
            num += len(self.F[u]['cpt'])
        return num

    def scope_size(self, rv):
        return len(self.F[rv]['parents'])+1

    def num_nodes(self):
        return len(self.V)

    def cpt(self, rv):
        return self.F[rv]['cpt']

    def card(self, rv):
        return len(self.F[rv]['values'])

    def scope(self, rv):
        scope = [rv]
        scope.extend(self.F[rv]['parents'])
        return scope

    def parents(self, rv):
        return self.F[rv]['parents']

    def children(self, rv):
        return self.E[rv]

    def degree(self, rv):
        return len(self.parents(rv)) + len(self.children(rv))

    def values(self, rv):
        return self.F[rv]['values']

    def value_idx(self, rv, val):
        try:   
            return self.F[rv]['values'].index(val)
        except ValueError:
            print("Value Index Error")
            return -1

    def stride(self, rv, n):
        if n==rv:
            return 1
        else:
            card_list = [self.card(rv)]
            card_list.extend([self.card(p) for p in self.parents(rv)])
            n_idx = self.parents(rv).index(n) + 1
            return int(np.prod(card_list[0:n_idx]))

    def flat_cpt(self, by_var=False, by_parents=False):
        """
        Return all cpt values in the BN as a flattened
        numpy array ordered by bn.nodes() - i.e. topsort
        """
        if by_var:
            cpt = np.array([sum(self.cpt(rv)) for rv in self.nodes()])
        elif by_parents:
            cpt = np.array([sum(self.cpt(rv)[i:(i+self.card(rv))]) for rv in self.nodes() for i in range(len(self.cpt(rv))/self.card(rv))])
        else:
            cpt = np.array([val for rv in self.nodes() for val in self.cpt(rv)])
        return cpt

    def cpt_indices(self, target, val_dict):
        """
        Get the index of the CPT which corresponds
        to a dictionary of rv=val sets. This can be
        used for parameter learning to increment the
        appropriate cpt frequency value based on
        observations in the data.

        There is definitely a fast way to do this.
            -- check if (idx - rv_stride*value_idx) % (rv_card*rv_stride) == 0

        Arguments
        ---------
        *target* : a string
            Main RV

        *val_dict* : a dictionary, where
            key=rv,val=rv value

        """
        stride = dict([(n,self.stride(target,n)) for n in self.scope(target)])
        #if len(val_dict)==len(self.parents(target)):
        #    idx = sum([self.value_idx(rv,val)*stride[rv] \
        #            for rv,val in val_dict.items()])
        #else:
        card = dict([(n, self.card(n)) for n in self.scope(target)])
        idx = set(range(len(self.cpt(target))))
        for rv, val in val_dict.items():
            val_idx = self.value_idx(rv,val)
            rv_idx = []
            s_idx = val_idx*stride[rv]
            while s_idx < len(self.cpt(target)):
                rv_idx.extend(range(s_idx,(s_idx+stride[rv])))
                s_idx += stride[rv]*card[rv]
            idx = idx.intersection(set(rv_idx))

        return list(idx)

    def cpt_str_idx(self, rv, idx):
        """
        Return string representation of RV=VAL and
        Parents=Val for the given idx of the given rv's cpt.
        """
        rv_val = self.values(rv)[idx % self.card(rv)]
        s = str(rv)+'='+str(rv_val) + '|'
        _idx=1
        for parent in self.parents(rv):
            for val in self.values(parent):
                if idx in self.cpt_indices(rv,{rv:rv_val,parent:val}):
                    s += str(parent)+'='+str(val)
                    if _idx < len(self.parents(rv)):
                        s += ','
                    _idx+=1
        return s



    def set_structure(self, edge_dict, value_dict=None):
        """
        Set the structure of a BayesNet object. This
        function is mostly used to instantiate a BN
        skeleton after structure learning algorithms.

        See "structure_learn" folder & algorithms

        Arguments
        ---------
        *edge_dict* : a dictionary,
            where key = rv,
            value = list of rv's children
            NOTE: THIS MUST BE DIRECTED ALREADY!

        *value_dict* : a dictionary,
            where key = rv,
            value = list of rv's possible values

        Returns
        -------
        None

        Effects
        -------
        - sets self.V in topsort order from edge_dict
        - sets self.E
        - creates self.F structure and sets the parents

        Notes
        -----

        """

        self.V = topsort(edge_dict)
        self.E = edge_dict
        self.F = dict([(rv,{}) for rv in self.nodes()])
        for rv in self.nodes():
            self.F[rv] = {
                'parents':[p for p in self.nodes() if rv in self.children(p)],
                'cpt': [],
                'values': []
            }
            if value_dict is not None:
                self.F[rv]['values'] = value_dict[rv]

    def adj_list(self):
        """
        Returns adjacency list of lists, where
        each list element is a vertex, and each sub-list is
        a list of that vertex's neighbors.
        """
        adj_list = [[] for _ in self.V]
        vi_map = dict((self.V[i],i) for i in range(len(self.V)))
        for u,v in self.edges():
            adj_list[vi_map[u]].append(vi_map[v])
        return adj_list

    def moralized_edges(self):
        """
        Moralized graph is the original graph PLUS
        an edge between every set of common effect
        structures -
            i.e. all parents of a node are connected.

        This function has be validated.

        Returns
        -------
        *e* : a python list of parent-child tuples.

        """
        e = set()
        for u in self.nodes():
            for p1 in self.parents(u):
                e.add((p1,u))
                for p2 in self.parents(u):
                    if p1!=p2 and (p2,p1) not in e:
                        e.add((p1,p2))
        return list(e)


    


    

def mle_estimator(bn, data, nodes=None, counts=False):
    """
    Maximum Likelihood Estimation is a frequentist
    method for parameter learning, where there is NO
    prior distribution. Instead, the frequencies/counts
    for each parameter start at 0 and are simply incremented
    as the relevant parent-child values are observed in the
    data. 

    This can be a risky method for small datasets, because if a 
    certain parent-child instantiation is never observed in the
    data, then its probability parameter will be ZERO (even if you
    know it should at least have a very small probability). 

    Note that the Bayesian and MLE estimators essentially converge
    to the same set of values as the size of the dataset increases.

    Also note that, unlike the structure learning algorithms, the
    parameter learning functions REQUIRE a passed-in BayesNet object
    because there MUST be some pre-determined structure for which
    we can actually learn the parameters. You can't learn parameters
    without structure - so structure must always be there first!

    Finally, note that this function can be used to calculate only
    ONE conditional probability table in a BayesNet object by
    passing in a subset of random variables with the "nodes"
    argument - this is mostly used for score-based structure learning,
    where a single cpt needs to be quickly recalculate after the
    addition/deletion/reversal of an arc.

    Arguments
    ---------
    *bn* : a BayesNet object
        The associated network structure for which
        the parameters will be learned

    *data* : a nested numpy array

    *nodes* : a list of strings
        Which nodes to learn the parameters for - if None,
        all nodes will be used as expected.

    Returns
    -------
    None

    Effects
    -------
    - modifies/sets bn.data to the learned parameters

    Notes
    -----
    - Currently doesn't return correct solution

    data attributes:
        "numoutcomes" : an integer
        "vals" : a list
        "parents" : a list or None
        "children": a list or None
        "cprob" : a nested python list

    - Do not want to alter bn.data directly!

    """
    if nodes is None:
        nodes = list(bn.nodes())
    else:
        if not isinstance(nodes, list):
            nodes = list(nodes)

    F = dict([(rv, {}) for rv in nodes])
    for i, n in enumerate(nodes):
        F[n]['values'] = list(np.unique(data[:,i]))
        bn.F[n]['values'] = list(np.unique(data[:,i]))

    obs_dict = dict([(rv,[]) for rv in nodes])
    # set empty conditional probability table for each RV
    for rv in nodes:
        # get number of values in the CPT = product of scope vars' cardinalities
        p_idx = int(np.prod([bn.card(p) for p in bn.parents(rv)])*bn.card(rv))
        F[rv]['cpt'] = [0]*p_idx
        bn.F[rv]['cpt'] = [0]*p_idx
    
    # loop through each row of data
    for row in data:
        # store the observation of each variable in the row
        for rv in nodes:
            obs_dict[rv] = row[rv]
        
        #obs_dict = dict([(rv,row[rv]) for rv in nodes])
        # loop through each RV and increment its observed parent-self value
        for rv in nodes:
            rv_dict= { n: obs_dict[n] for n in obs_dict if n in bn.scope(rv) }
            offset = bn.cpt_indices(target=rv,val_dict=rv_dict)[0]
            F[rv]['cpt'][offset]+=1

    if counts:
        return F
    else:
        for rv in nodes:
            F[rv]['parents'] = [var for var in nodes if rv in bn.E[var]]
            for i in range(0,len(F[rv]['cpt']),bn.card(rv)):
                temp_sum = float(np.sum(F[rv]['cpt'][i:(i+bn.card(rv))]))
                for j in range(bn.card(rv)):
                    F[rv]['cpt'][i+j] /= (temp_sum+1e-7)
                    F[rv]['cpt'][i+j] = round(F[rv]['cpt'][i+j],5)
        bn.F = F


def log_likelihood(bn, nrow):
    """
    Determining log-likelihood of the parameters
    of a Bayesian Network. This is a quite simple
    score/calculation, but it is useful as a straight-forward
    structure learning score.

    Semantically, this can be considered as the evaluation
    of the log-likelihood of the data, given the structure
    and parameters of the BN:
        - log( P( D | Theta_G, G ) )
        where Theta_G are the parameters and G is the structure.

    However, for computational reasons it is best to take
    advantage of the decomposability of the log-likelihood score.
    
    As an example, if you add an edge from A->B, then you simply
    need to calculate LOG(P'(B|A)) - Log(P(B)), and if the value
    is positive then the edge improves the fitness score and should
    therefore be included. 

    Even more, you can expand and manipulate terms to calculate the
    difference between the new graph and the original graph as follows:
        Score(G') - Score(G) = M * I(X,Y),
        where M is the number of data points and I(X,Y) is
        the marginal mutual information calculated using
        the empirical distribution over the data.

    In general, the likelihood score decomposes as follows:
        LL(D | Theta_G, G) = 
            M * Sum over Variables ( I ( X , Parents(X) ) ) - 
            M * Sum over Variables ( H( X ) ),
        where 'I' is mutual information and 'H' is the entropy,
        and M is the number of data points

    Moreover, it is clear to see that H(X) is independent of the choice
    of graph structure (G). Thus, we must only determine the difference
    in the mutual information score of the original graph which had a given
    node and its original parents, and the new graph which has a given node
    and new parents.

    NOTE: This assumes the parameters have already
    been learned for the BN's given structure.

    LL = LL - f(N)*|B|, where f(N) = 0

    Arguments
    ---------
    *bn* : a BayesNet object
        Must have both structure and parameters
        instantiated.
    Notes
    -----
    NROW = data.shape[0]
    mi_score = 0
    ent_score = 0
    for rv in bn.nodes():
        cols = tuple([bn.V.index(rv)].extend([bn.V.index(p) for p in bn.parents(rv)]))
        mi_score += mutual_information(data[:,cols])
        ent_score += entropy(data[:,bn.V.index(rv)])
    
    return NROW * (mi_score - ent_score)
    """
    return np.sum(np.log(nrow * (bn.flat_cpt()+1e-7)))

def MDL(bn, nrow):
    """
    Minimum Description Length score - it is
    equivalent to BIC
    """
    return BIC(bn, nrow)

def BIC(bn, nrow):
    """
    Bayesian Information Criterion.

    BIC = LL - f(N)*|B|, where f(N) = log(N)/2

    """
    log_score = log_likelihood(bn, nrow)
    penalty = 0.5 * bn.num_params() * np.log(max(bn.num_edges(),1))
    return log_score - penalty

def AIC(bn, nrow):
    """
    Aikaike Information Criterion

    AIC = LL - f(N)*|B|, where f(N) = 1

    """
    log_score = log_likelihood(bn, nrow)
    penalty = len(bn.flat_cpt())
    return log_score - penalty



def info_score(bn, nrow, metric='BIC'):
    if metric.upper() == 'LL':
        score = log_likelihood(bn, nrow)
    elif metric.upper() == 'BIC':
        score = BIC(bn, nrow)
    elif metric.upper() == 'AIC':
        score = AIC(bn, nrow)
    else:
        score = BIC(bn, nrow)

    return score
    

def unique_bins(data):
    """
    Get the unique values for each column in a dataset.
    """
    bins = np.empty(len(data.T), dtype=np.int32)
    i = 0
    for col in data.T:
        bins[i] = len(np.unique(col))
        i+=1
    return bins



import networkx as nx
import numpy as np
from copy import copy

def would_cause_cycle(e, u, v, reverse=False):
    """
    Test if adding the edge u -> v to the BayesNet
    object would create a DIRECTED (i.e. illegal) cycle.
    """
    G = nx.DiGraph(e)
    if reverse:
        G.remove_edge(v,u)
    G.add_edge(u,v)
    try:
        nx.find_cycle(G, source=u)
        return True
    except:
        return False
