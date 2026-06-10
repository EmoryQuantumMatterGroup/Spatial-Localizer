import numpy as np

from .matrices import *


class Localizer() :
    
    def __init__(self, operators, p_spaces, clifford_elements=None, init_param=dict()) :
        """Inititalization function for a general localizer.

        Args:
            operators (list): List of operators as ndarrays to localize. Must be all Unitary or all Hermitian at the moment
            p_spaces (_type_): list of parameter spaces as ndarrays for each operator. The shape of p_spaces[i] must be equal for all i
            init_param (dict, optional): _description_. Defaults to empty dictionary.
        """
        
        self.init_param = {
        "H_or_U" : "U",
        "enforce_unitarity" : True,
        "unitary_real_embed_style" : "separate", # options are together or separate 
        "unitary_scan_style" : "rotate", # options are rotate or translate
        "print_unitarity_metrics" : False
        }   
        
        for key in init_param.keys() : 
            self.init_param[key] = init_param[key]
        
        self.operators = operators
        self.p_spaces = p_spaces
        self.clifford_elements = clifford_elements
        
        self.num_ops = len(self.operators)
        # get shape of p_spaces 
        self.p_shape = np.shape(self.p_spaces[0]) # need for reshaping
        self.op_dim = len(self.operators[0]) # should be square operators

            
            
        match self.init_param["H_or_U"] :
            
            case "H" :
                self.hermitian_init(self.operators)
            
            case "U" :
                self.unitary_init(self.operators)
            
            case _ : 
                raise ValueError("Incorrect parameter provided. Options are 'H' or 'U'")
        
        self.cliff_dim = len(self.clifford_elements[0]) # should be square operators
            
            
    def hermitian_init(self, operators) : 
        
        if type(self.clifford_elements) == type(None) : 
            if len(operators) <= 3 :
                self.clifford_elements = get_pauli("xyz")
                
            elif len(operators) <= 5 : 
                self.clifford_elements = get_gamma("all")
            
            else : 
                raise ValueError("""Too many operators for the default available clifford elements. 
                                 Please provide your own list of clifford elements through the initalization variable 'clifford_elements'. This software can handle at most 5 hermitian operators or 4 unitaries using stock clifford elements.""")
        else :       
            if len(operators) > len(self.clifford_elements) : 
                raise ValueError("Too may operators for provided clifford elements. Must have as many clifford elements as operators")
            
        self.cliff_dim = len(self.clifford_elements[0])
            

                
    def unitary_init(self, operators) : 
        
        
        if self.init_param["unitary_real_embed_style"] == "together" : 
            if type(self.clifford_elements) == type(None) : 
                if len(operators) <= 2 :
                    self.clifford_elements = get_pauli("xyz")

                    
                elif len(operators) <= 4 : 
                    self.clifford_elements = get_gamma("all")
                
                else : 
                    raise ValueError("""Too many operators for the default available clifford elements. 
                                    Please provide your own tuple of clifford elements through the initalization variable 'clifford_elements'. This software can handle at most 5 hermitian operators or 2 unitaries using stock clifford elements.""")                

            else :       
                if len(operators) > len(self.clifford_elements) - 1 : 
                    raise ValueError("Too may operators for provided clifford elements. Must have at least one more clifford elements than number of operators for unitary localizers.")

            
            # breaking unitaries into real and imaginary parts
                
            if self.init_param["enforce_unitarity"] : # enforcing unitarity if desired
                for ii, op in enumerate(operators) : 
                    
                    if self.init_param["print_unitarity_metrics"] : print("Enforcing unitarity of operator, printing metrics below.") 
                    operators[ii] = enforce_unitarity(op,self.init_param["print_unitarity_metrics"])
        
        elif self.init_param["unitary_real_embed_style"] == "separate" : 
            if type(self.clifford_elements) == type(None) : 
                if len(operators) <= 1 :
                    self.clifford_elements = get_pauli("xyz")

                    
                elif len(operators) <= 2 : 
                    self.clifford_elements = get_gamma("all")
                
                else : 
                    raise ValueError("""Too many operators for the default available clifford elements. 
                                    Please provide your own tuple of clifford elements through the initalization variable 'clifford_elements'. This software can handle at most 5 hermitian operators or 4 unitaries using stock clifford elements.""")                

            else :       
                if len(operators) > len(self.clifford_elements) - 1 : 
                    raise ValueError("Too may operators for provided clifford elements. Must have at least one more clifford elements than number of operators for unitary localizers.")

            
            # breaking unitaries into real and imaginary parts
                
            if self.init_param["enforce_unitarity"] : # enforcing unitarity if desired
                for ii, op in enumerate(operators) : 
                    
                    if self.init_param["print_unitarity_metrics"] : print("Enforcing unitarity of operator, printing metrics below.") 
                    operators[ii] = enforce_unitarity(op,self.init_param["print_unitarity_metrics"])
        
    
    def build_locs(self) :
        
        match self.init_param["H_or_U"] :
            
            case "H" :
                raise ValueError("This method has not been implemented")
            
            case "U" : 
                
                match self.init_param["unitary_scan_style"] : 
                    
                    case "rotate" :
                        self.locs = self._build_locs_unitary_rotate(self.operators,self.p_spaces,self.clifford_elements,self.init_param["unitary_real_embed_style"])
                        
                    case "translate" :
                        raise ValueError("This method has not been implemented")
                    
                    case _ : 
                        raise ValueError("Invalid option for 'unitary_scan_style'. Options are 'translate' and 'rotate'")
                    
            case _ :
                raise ValueError("Invalid option for 'H_or_U'. Options are 'H' and 'U'")
                
        
    def _build_locs_unitary_rotate(self, operators, p_spaces, clifford_elements, embed_style="together") : 
        
        # tensor p_spaces with respective operators
        
        ops_tensored = [] 
        
        for ii in range(len(operators)) : 
            ops_tensored.append(np.tensordot(np.conj(p_spaces[ii].flat),operators[ii],axes=0)) # implements op @ (np.exp(-p_space*identity))
            
        eye = np.tensordot(np.ones(np.shape(p_spaces[0].flat),dtype=complex),np.eye(len(operators[0]), dtype=complex),axes=0) # identity to work with
            
        # split operators into real and imaginary parts
        op_real = []
        op_imag = [] 
        
        for op in ops_tensored :
            op_r, op_i = unt_to_real_imag(op)
            
            op_real.append(op_r)
            op_imag.append(op_i)
            
        
        # embedding operators into clifford elements 
        
        if embed_style == "together" : # embed real components into same clifford element
            
            for ii in range(self.num_ops) : 
                if ii == 0 : 
                    locs = np.kron(op_real[ii] - eye,clifford_elements[0])
                    locs += np.kron(op_imag[ii],clifford_elements[ii+1])
                else : 
                    locs += np.kron(op_real[ii] - eye,clifford_elements[0])
                    locs += np.kron(op_imag[ii],clifford_elements[ii+1])
                
        elif embed_style == "separate" : # embed real components into different clifford elements
            
            # clifford_elements = list(mm.get_gamma("all"))
            # self.cliff_dim=4
            # print("changing cliff dim")
            if len(operators) > 2*len(clifford_elements)  : 
                raise ValueError("Too many operators for provided clifford elements. Must have at least twice as many clifford elements than number of operators for unitary localizers with each real/imaginary component having it's own component.")
            
            for ii in range(self.num_ops) : 
                if ii == 0 : 
                    locs = np.kron(op_real[ii] - eye,clifford_elements[2*ii])
                    locs += np.kron(op_imag[ii],clifford_elements[2*ii+1])
                else : 
                    locs += np.kron(op_real[ii] - eye,clifford_elements[2*ii])
                    locs += np.kron(op_imag[ii],clifford_elements[2*ii+1])
        else : 
            raise ValueError("Invalid embed style has been given. Options are 'together' and 'separate'")
        
        return locs
    
    def scan(self, num_keep=1) : 
        
        self.build_locs()
        
        return self._scan(self.locs, self.p_shape, num_keep=num_keep)
        
    
    def _scan(self, locs, p_shape, num_keep=1) : 
        
        if num_keep==1 : 
            return self._get_pseudospectrum(locs,p_shape)
        else :
            ws = np.linalg.eigvalsh(locs)
            
            indices = np.argsort(np.abs(ws),axis=-1)
            
            ws = np.take_along_axis(ws,indices,axis=-1) # sorting ws
            
            return np.reshape(ws[:,:num_keep],(*p_shape,num_keep)) 
        
    
    def get_operator_eigenvectors(self,num_keep=1,filter_method="raw",ifNormalize=True) :
        
        return self._get_operator_eigenvectors(self.p_spaces,self.cliff_dim,num_keep=num_keep, filter_method=filter_method,ifNormalize=ifNormalize)
    
    def _get_operator_eigenvectors(self, p_spaces, cliff_dim, num_keep=1, filter_method="raw",ifNormalize=True) :
        
        print(f"new cliff dim is {self.cliff_dim}")
        
        
        loc_vecs = self._get_loc_vecs(p_spaces,num_keep)
        
        loc_dim, num_vecs = loc_vecs.shape 
        
        
        match filter_method :
            
            case "norm" :
                
                op_vecs = np.zeros((loc_dim//cliff_dim, num_vecs),dtype=complex)
                
                for ii in range(num_vecs) : 
                    loc_vec = loc_vecs[:,ii]
                    
                    norms = [] # collecting norms of vectors
                    
                    for jj in range(cliff_dim) :
                        temp_vec = loc_vec[jj::cliff_dim]
                        norms.append(np.linalg.norm(temp_vec))
                    if ifNormalize :
                        op_vecs[:,ii] = loc_vec[np.argmax(norms)::cliff_dim]/np.linalg.norm(loc_vec[np.argmax(norms)::cliff_dim])
                    else : 
                        op_vecs[:,ii] = loc_vec[np.argmax(norms)::cliff_dim]
                    
                return op_vecs
                    
            case 'raw' : 
                raw_vecs = np.zeros((loc_dim//cliff_dim, num_vecs, cliff_dim), dtype=complex)
                norms = np.zeros((num_vecs, cliff_dim), dtype=complex)
                for ii in range(num_vecs) : 
                    loc_vec = loc_vecs[:,ii]
                    for jj in range(cliff_dim) :
                        
                        if ifNormalize :
                            raw_vecs[:,ii,jj] = loc_vec[jj::cliff_dim]/np.linalg.norm(loc_vec[jj::cliff_dim])
                        else : 
                            raw_vecs[:,ii,jj] = loc_vec[jj::cliff_dim]
                        
                        norms[ii,jj] = np.linalg.norm(loc_vec[jj::cliff_dim])
                        
                        
                        
                return raw_vecs, norms
                    
            
            case _ :
                raise ValueError("Invalid filter method provided. options are 'norm' and 'raw'")
            
            
    def get_loc_vecs(self,num_keep=1) :
        
        return self._get_loc_vecs(self.p_spaces,num_keep=num_keep)
        
    def _get_loc_vecs(self, p_spaces, num_keep=1) :
        
        vecs = []
        
        
        locs = self._build_locs_unitary_rotate(self.operators,p_spaces,self.clifford_elements,self.init_param["unitary_real_embed_style"])
        
        ws_all, vs_all = np.linalg.eigh(locs)
    
        
        for ii in range(len(p_spaces[0].flat)) : # slow way, need to build faster way
            
            ws = ws_all[ii]
            vs = vs_all[ii]
            
            sorted_indices = np.argsort(np.abs(ws))
            sorted_eigenvectors = vs[:, sorted_indices]
            
            for jj in range(num_keep) : 
                vecs.append(sorted_eigenvectors[:,jj])
                
        return np.array(vecs).transpose()
    
        
    def _get_pseudospectrum(self, locs, p_shape) :
        
        return np.reshape(np.min(np.abs(np.linalg.eigvalsh(locs)),axis=-1),p_shape)
             
        
            
        
        
        
            
        
        
        
            
        
            
            

            
        
        
            
            
                
            
                
        
                
                
        
        
        
        
        
                
        