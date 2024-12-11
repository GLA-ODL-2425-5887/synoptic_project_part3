import pytest
from multipartite.base import Multipartite 

def test_makeGraph_simple():
    # simple tests to check that we can make a graph 
    # without an error
    # would not be a usual test but want a pass
    # when students first pull the repo 
    test_G = Multipartite(3)

def test_makeGraph_fail():
    with pytest.raises(ValueError):
        test_G = Multipartite(None)
    with pytest.raises(ValueError):
        test_G = Multipartite(1.0)
    with pytest.raises(ValueError):
        test_G = Multipartite("squirrel")
    with pytest.raises(ValueError):
        test_G = Multipartite(-1)

def test_add_node():
    for idx in range(2,11):
        test_G = Multipartite(idx)
        nodes = []
        for i in range(idx):
            nodes.append(test_G.add_node(group=i))
        nodesG = test_G.get_nodes()   
        assert len(nodes) == len(nodesG)
        for node in nodes:
            assert node in nodesG


def test_add_node_testFail():
    for i in range(1,100):
        test_G = Multipartite(i)
        for k in range(10):
            with pytest.raises(Exception):
                test_G.add_node(i+k)


def test_add_edge_test1():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(1)
    assert n1 != n2
    test_G.add_multipartite_edge(n1,n2)
    ed = test_G.get_edges()
    assert len(ed)==1
    assert n1 in ed[0]
    assert n2 in ed[0]

def test_add_edge_test2():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(2)
    assert n1 != n2
    test_G.add_multipartite_edge(n1,n2)
    ed = test_G.get_edges()
    assert len(ed)==1
    assert n1 in ed[0]
    assert n2 in ed[0]

def test_add_edge_test3():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(4)
    test_G.add_multipartite_edge(n1,n2)
    ed = test_G.get_edges()
    assert len(ed)==1
    assert n1 in ed[0]
    assert n2 in ed[0]

def test_add_edge_fail_on_bad_node1():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(1)

    n1 = str(n1)+"fake"
    with pytest.raises(Exception):
        test_G.add_multipartite_edge(n1,n2)

def test_add_edge_fail_on_bad_node2():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(1)

    n1 = None 
    with pytest.raises(Exception):
        test_G.add_multipartite_edge(n1,n2)

def test_add_edge_fail_on_bad_node3():
    test_G = Multipartite(5)
    n1 = test_G.add_node(0)
    n2 = test_G.add_node(1)

    n2 = None 
    with pytest.raises(Exception):
        test_G.add_multipartite_edge(n1,n2)
