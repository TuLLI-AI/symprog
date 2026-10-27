# type: ignore
import pytest

from symprog import Parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prob(program) -> float:
    return program.inference().inference.item()

# ---------------------------------------------------------------------------
# Comparison to problog: https://dtai.cs.kuleuven.be/problog/tutorial/basic/10_inhibitioneffects.html
# ---------------------------------------------------------------------------

def test_inhibition_basic():
    # https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=e5970d5443c8e62c298a46a87fa29689
    @Parser(language="plp")
    def noisyor():
        c1: 0.5; c2: 0.5
        
        e[0.3] :- c1 
        e[0.4] :- c2
        
        evidence(c1, True)
        evidence(c2, True)
        
        query(e)

    assert prob(noisyor) == pytest.approx(0.5800000429153442, abs=1e-5)

def test_inhibition_1(): 
    @Parser(language="plp")
    def inhibition_1():
        p1: 0.4; p2: 0.75
        n1: 0.3; n2: 0.6
        
        p[0.6] :- p1 
        p[0.7] :- p2
        n[0.7] :- n1
        n[0.6] :- n2
        y :- p
        y[False] :- n
        
        ## evidence and query 
        evidence(p1, False)
        evidence(n1, False)
        query(y)

    assert prob(inhibition_1) == pytest.approx(0.3359999656677246, abs=1e-5)

def test_inhibition_2():
    # https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=ecfaa78c4e594401b76f230f45a64f8c
    @Parser(language="plp")
    def inh():
        c1: 0.5; c2: 0.5
            
        e1[0.3] :- c1 
        ## weighted inhibitory rules 
        e1[0.2, False] :- c2

        e2[0.4] :- c2 
        ## weighted inhibitory rules 
        e2[0.2, False] :- c1

        e :- e1 
        e :- e2
        
        evidence(c1, True)
        evidence(c2, True)
            
        query(e)

    assert prob(inh) == pytest.approx(0.4832, abs=1e-5)

def test_inhibition_3():
    # https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=ecfaa78c4e594401b76f230f45a64f8c
    @Parser(language="plp")
    def inh2():
        c1: 0.5; c2: 0.5
            
        e[0.3] :- c1 
        e[0.4] :- c2 

        e[0.166666, False] :- c1 & c2
        evidence(c1, True)
        evidence(c2, True)
        query(e)

    assert prob(inh2) == pytest.approx(0.4833333, abs=1e-5)

def test_non_disjoint_proof():
    # https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=cdcaf69e24a6d1979c6853987bf25096
    @Parser(language="plp")
    def disjoint():
        head_1: 0.4; head_2: 0.7; head_3: 0.5
        win :- head_1
        win :- head_2 & head_3 
        query(win)

    assert prob(disjoint) == pytest.approx(0.61, abs=1e-5)

def test_ad_1():
    @Parser(language="plp")
    def simple_ad():
        a[0.5] | c[0.05]
        f:0.4
        
        b :- a 
        b :- c
        d :- b & f

        query(b)

    assert prob(simple_ad) == pytest.approx(0.55000, abs=1e-5)    


def test_bayes_net_1():
    
    @Parser(language="plp")
    def bayes1():
        burglary: 0.7
        earthquake: 0.2
        p_alarm1: 0.9
        p_alarm2: 0.8
        p_alarm3: 0.1

        alarm :- burglary & earthquake & p_alarm1
        alarm :- burglary & ~earthquake & p_alarm2
        alarm :- ~burglary & earthquake & p_alarm3

        evidence(alarm, True)
        query(burglary)

    assert prob(bayes1) == pytest.approx(0.98965517, abs=1e-5)

def test_bayes_net_2():
    
    @Parser(language="plp")
    def bayes2():
        burglary: 0.7
        earthquake: 0.2
        p_alarm1: 0.9
        p_alarm2: 0.8
        p_alarm3: 0.1

        alarm :- burglary & earthquake & p_alarm1
        alarm :- burglary & ~earthquake & p_alarm2
        alarm :- ~burglary & earthquake & p_alarm3

        evidence(alarm, True)
        query(earthquake)

    assert prob(bayes2) == pytest.approx(0.22758621, abs=1e-5)

def test_bayes_net_3():
    
    @Parser(language="plp")
    def bayes2():
        burglary: 0.7
        earthquake: 0.2

        alarm[0.9] :- burglary & earthquake
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake

        evidence(alarm, True)
        query(earthquake)

    assert prob(bayes2) == pytest.approx(0.22758621, abs=1e-5)

def test_ad(): 
    @Parser(language="plp")
    def ad_1():
        a[0.5] | c[0.05]
        f:0.4
        
        b :- a 
        b :- c
        d :- b & f

        evidence(a, True)
        query(b) 

    assert prob(ad_1) == pytest.approx(1.0, abs=1e-5)

def test_ad_2(): 
    @Parser(language="plp")
    def ad_2():
        a[0.5] | c[0.05] 
        f:0.4
        
        b :- a 
        b :- c
        d :- b & f

        evidence(a, False)
        query(b) 

    assert prob(ad_2) == pytest.approx(0.1, abs=1e-5)

def test_first_order_1():
    @Parser(language="plp")
    def fo():
        person[john] : 0.5; person[mary] : 0.7

        burglary: 0.7 
        earthquake: 0.2 

        alarm[0.9] :- burglary & earthquake 
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake

        calls[X, 0.8] :- alarm & person[X]
        calls[X, 0.1] :- ~alarm & person[X]

        query(calls[john])    

    assert prob(fo) == pytest.approx(0.25299999, abs=1e-5)

def test_first_order_2():
    @Parser(language="plp")
    def fo():
        person[john] : 1.0; person[mary] : 1.0

        burglary: 0.7; earthquake: 0.2 

        alarm[0.9] :- burglary & earthquake 
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake

        calls[X, 0.8] :- alarm & person[X]
        calls[X, 0.1] :- ~alarm & person[X]

        query(calls[john])    

    assert prob(fo) == pytest.approx(0.506, abs=1e-5)

# def test_graph_1(): 
#     # https://dtai.cs.kuleuven.be/problog/tutorial/basic/04_pgraph.html
#     @Parser(language="plp")
#     def graph():
#         edge[n1,n2]: 0.6
#         edge[n1,n3]: 0.1 
#         edge[n2,n5]: 0.4
#         edge[n2,n6]: 0.3
#         edge[n3,n4]: 0.3 
#         edge[n4,n5]: 0.8
#         edge[n5,n6]: 0.2 

#         path[X,Y] :- edge[X,Y]
#         path[X,Y] :- edge[X,Z] & path[Z,Y]

#         query(path[n1,n6])   

#     assert prob(graph) == pytest.approx(0.2167296, abs=1e-5)

# def test_graph_2(): 
#     # https://dtai.cs.kuleuven.be/problog/tutorial/basic/04_pgraph.html
#     @Parser(language="plp")
#     def graph():
#         edge[n1,n2]: 0.6
#         edge[n1,n3]: 0.1 
#         edge[n2,n5]: 0.4
#         edge[n2,n6]: 0.3
#         edge[n3,n4]: 0.3 
#         edge[n4,n5]: 0.8
#         edge[n5,n6]: 0.2 

#         path[X,Y] :- edge[X,Y]
#         path[X,Y] :- edge[X,Z] & path[Z,Y]

#         query(path[n1,n5])   

#     assert prob(graph) == pytest.approx(0.25824, abs=1e-5)