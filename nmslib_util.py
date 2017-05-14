import numpy as np
import nmslib 
import pickle
import numbers

class CBIMIndexSpaceTypes:
    L1='l1'
    L2='l2'
    ANGULAR_DISTANCE='angulardist'
    COSINE_SIMILARITY='cosinesimil'
    DEFAULT=COSINE_SIMILARITY
    
class CBIMIndexSpaceParam:
    KNN=['knn=1']
    RANGE=['range=1.0']
    BOTH=['knn=1','range=1.0']
    DEFAULT=KNN
    
class CBIMIndexMethodNames:
    VP_TREE='vptree'
    MVP_TREE='mvptree'
    GH_TREE='ghtree'
    LIST_OF_CLUSTERS='list_clusters'
    SA_TREE='satree'
    BB_TREE='bbtree'
    SW_GRAPH='sw-graph'
    HIERARCHICAL_NAVIGABLE_SW_GRAPH='hnsw'
    NN_DESCENT='nndes'
    DEFAULT=VP_TREE
    
def index_param_factory(method_name):

    if method_name is CBIMIndexMethodNames.VP_TREE:
        return {
            'bucketSize':10,
            'chunkBucket':1,
            'selectPivotAttempts':5
        }
    if method_name is CBIMIndexMethodNames.MVP_TREE:
        return {
            'bucketSize':10,
            'chunkBucket':1,
            'maxPathLen':20
        }
    if method_name is CBIMIndexMethodNames.GH_TREE:
        return {
            'bucketSize':10,
            'chunkBucket':1
        }
    if method_name is CBIMIndexMethodNames.LIST_OF_CLUSTERS:
        return {
            'bucketSize':10,
            'chunkBucket':1,
            'useBucketSize':0,
            'radius':0.5,
            'strategy':'random' # can_be: 'random','closestPrevCenter','farthestPrevCenter','minSumDistPrevCenters','maxSumDistPrevCenters'

            }
    if method_name is CBIMIndexMethodNames.SA_TREE:
        return {}
    if method_name is CBIMIndexMethodNames.BB_TREE:
        return {            
            'bucketSize':10,
            'chunkBucket':1
        }
    if method_name is CBIMIndexMethodNames.SW_GRAPH:
        return {            
            'NN':3,
            'initIndexAttempts':5,
            'indexThreadQty':4
        }
    if method_name is CBIMIndexMethodNames.HIERARCHICAL_NAVIGABLE_SW_GRAPH:
        return {            
            'M':10,
            'efConstruction':20,
            'indexThreadQty':4,
            'searchMethod':5
        }
    if method_name is CBIMIndexMethodNames.NN_DESCENT:
        return {            
            'NN':10,
            'rho':0.5,
            'delta':0.001
        }
    return {}

def query_time_param_factory(method_name):
    if method_name is CBIMIndexMethodNames.VP_TREE:
        return {
            'alphaLeft':2.0,
            'alphaRight':2.0,
            'expLeft':1,
            'expRight':1,
            'maxLeavesToVisit':2147483647
        }
    if method_name is CBIMIndexMethodNames.MVP_TREE:
        return {}
    if method_name is CBIMIndexMethodNames.GH_TREE:
        return {}
    if method_name is CBIMIndexMethodNames.LIST_OF_CLUSTERS:
        return {}
    if method_name is CBIMIndexMethodNames.SA_TREE:
        return {}
    if method_name is CBIMIndexMethodNames.BB_TREE:
        return {}
    if method_name is CBIMIndexMethodNames.SW_GRAPH:
        return {            
            'initSearchAttempts':1,
            'efSearch':10
        }
    if method_name is CBIMIndexMethodNames.HIERARCHICAL_NAVIGABLE_SW_GRAPH:
        return {            
            'efSearch':10
        }
    if method_name is CBIMIndexMethodNames.NN_DESCENT:
        return {            
            'initSearchAttempts':3
        }
    return {}


def  params_to_list(index_param):
    params=[]
    for key, value in index_param.iteritems():
        params.append(str(key)+'='+str(value))
    return params
    
def print_params(index_param):
    for key, value in index_param.iteritems():
        print str(key)+'='+str(value)
        

class CBIMConf:
    '''
    Config file for index creation.
    '''
    space_type = None
    space_param = None
    method_name = None
    index_param = None
    query_time_param = None
    
    def __init__(self,space_type=CBIMIndexSpaceTypes.DEFAULT,
                 space_param=CBIMIndexSpaceParam.DEFAULT,
                 method_name=CBIMIndexMethodNames.DEFAULT):          
        self.space_type = space_type
        self.space_param = space_param
        self.method_name = method_name
        self.index_param = params_to_list(index_param_factory(method_name))
        self.query_time_param =  params_to_list(query_time_param_factory(method_name))  

class CBIMIndex:
    index=None
    conf=None
    created=False
    
    def __init__(self,cbim_conf):
        self.conf=cbim_conf
        print cbim_conf.space_type
        print cbim_conf.space_param
        print cbim_conf.method_name
        
        self.index = nmslib.init(cbim_conf.space_type,
                        cbim_conf.space_param,
                        cbim_conf.method_name,
                        nmslib.DataType.DENSE_VECTOR,
                        nmslib.DistType.FLOAT)

    def add_data(self,data,indexes=None,batch_size=100,convert=False):
        if self.created:
            return -1
        if data.dtype is not 'float32':
            if not convert:
                print 'WARNING: Given data is not of type float32.'
                print '\tEnable conversion with flag convert=True to cast data or give float32 data for nmslib to work properly.'
                return -1
            else:
                data=data.astype(np.float32)
        return self._add_data_to_index(data,indexes,batch_size)
    
    def _add_data_to_index(self,data,indexes,batch_size):
        if type(indexes) is list:
            offset=indexes[0]
        elif isinstance(indexes, numbers.Number):
            offset=indexes
        else:
            offset=0 
        for data_batch in np.split(data,batch_size,axis=0):
            indices=np.arange(len(data_batch),dtype=np.int32)+offset
            nmslib.addDataPointBatch(self.index,indices,data_batch)
            offset+=data_batch.shape[0]
        return offset

    
    def create(self):
        if self.created:
            return False
        else:
            nmslib.createIndex(self.index, self.conf.index_param)
            nmslib.setQueryTimeParams(self.index,self.conf.query_time_param)
            self.created=True
            return True
    
    def query(self,query_data,top_k=3,convert=False,num_threads = 4):
        if not self.created:
            return -1
        if query_data.dtype is not 'float32':
            if not convert:
                print 'WARNING: Given query data is not of type float32.'
                print '\tEnable conversion with flag convert=True to cast query data or give float32 query data for nmslib to work properly.'
                return -1
            else:
                query_data=query_data.astype(np.float32)
        
        return nmslib.knnQueryBatch(self.index, num_threads, top_k, query_data)
    
    def pickle(self,path):
        pickle.dump(self, open(path,'wb'))
    
    def clr_mem(self):
        nmslib.freeIndex(self.index)
        self.created=False
        
def CBIMIndex_unpickle(path):
    return pickle.load(open(path,'rb'))
        
def create_index(data,conf=None,batch_size=5,convert=False):
    '''
    NMSLIB knows how to work only with 32bit data, so a conversion is needed for it to work properly. 
    '''
    if conf is None:
        conf=CBIMConf()
    index=CBIMIndex(conf)
    status=index.add_data(data,batch_size=batch_size,convert=convert)
    if status is not -1:
        index.create()
        return index
    else:
        return None
    
def query_index(index,query,top_k=3,convert=False):
    return index.query(query_data=query,top_k=top_k,convert=convert)


