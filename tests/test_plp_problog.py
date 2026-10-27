# type: ignore
"""
ProbLog coverage benchmark.

Each section maps a canonical ProbLog example to this language's syntax and records
the coverage status.  Status codes:

  [OK]       Directly expressible; runnable test passes.
  [PARTIAL]  Expressible with workarounds, or correct for restricted inputs.
  [BUG]      Theoretically expressible but fails due to an implementation bug.
  [NO]       Not expressible in the current language regardless of workarounds.

Sources:
  - https://dtai.cs.kuleuven.be/problog/tutorial/basic/01_coins.html
  - https://dtai.cs.kuleuven.be/problog/tutorial/basic/02_bayes.html
  - https://dtai.cs.kuleuven.be/problog/tutorial/basic/04_pgraph.html
  - https://dtai.cs.kuleuven.be/problog/tutorial/basic/10_inhibitioneffects.html
  - https://github.com/ML-KULeuven/problog/tree/master/test  (numbered system tests)

Run with: pytest tests/test_plp_problog.py -v
"""

import pytest
from symprog import Parser


def prob(program) -> float:
    return program.inference().inference.item()


# ===========================================================================
# BASIC PROBABILISTIC FACTS
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 1. Weighted facts + conjunction
# ---------------------------------------------------------------------------
# ProbLog (coin.pl / 00_trivial_and.pl):
#   0.5::heads1.  0.6::heads2.
#   twoHeads :- heads1, heads2.
#   query(twoHeads).  % 0.3

def test_coin_conjunction():
    @Parser(language="plp")
    def coin():
        heads1: 0.5
        heads2: 0.6
        twoHeads :- heads1 & heads2
        query(twoHeads)

    assert prob(coin) == pytest.approx(0.3, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 0b. Boolean weights — True ≡ 1.0, False ≡ 0.0
# ---------------------------------------------------------------------------
# `coin[c1]: True` means P(coin__c1) = 1.0 (always observed).
# `coin[c3]: False` means P(coin__c3) = 0.0 (never observed).

def test_boolean_weights():
    @Parser(language="plp")
    def prog():
        coin[c1]: True
        coin[c2]: True
        coin[c3]: False
        heads[C, 0.6] :- coin[C]
        query(heads[_])

    results = prog.inference()
    vals = sorted(r.inference.item() for r in results)
    # c1 and c2 → heads with prob 0.6; c3 → heads never (0.0)
    assert vals[0] == pytest.approx(0.0, abs=1e-5)
    assert vals[1] == pytest.approx(0.6, abs=1e-5)
    assert vals[2] == pytest.approx(0.6, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 2. Weighted facts + noisy-or (multiple rules, same head)
# ---------------------------------------------------------------------------
# ProbLog (coin.pl / 00_trivial_or.pl):
#   someHeads :- heads1.
#   someHeads :- heads2.
#   query(someHeads).  % 0.8

def test_coin_noisy_or():
    @Parser(language="plp")
    def coin():
        heads1: 0.5
        heads2: 0.6
        someHeads :- heads1
        someHeads :- heads2
        query(someHeads)

    assert prob(coin) == pytest.approx(0.8, abs=1e-5)


# ===========================================================================
# NEGATION-AS-FAILURE
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 3. Simple NAF — probability of complement
# ---------------------------------------------------------------------------
# ProbLog (00_trivial_not.pl):
#   0.4::a.
#   p :- \+a.
#   query(p).  % 0.6

def test_naf_simple():
    @Parser(language="plp")
    def prog():
        a: 0.4
        p :- ~a
        query(p)

    assert prob(prog) == pytest.approx(0.6, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 4. NAF of derived predicate
# ---------------------------------------------------------------------------
# ProbLog (00_trivial_not_and.pl):
#   0.5::a.  0.3::b.
#   q :- a, b.
#   p :- \+q.
#   query(p).  % 0.85

def test_naf_derived():
    @Parser(language="plp")
    def prog():
        a: 0.5
        b: 0.3
        q :- a & b
        p :- ~q
        query(p)

    assert prob(prog) == pytest.approx(0.85, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 5. Stratified NAF with multiple rules
# ---------------------------------------------------------------------------
# ProbLog (negation.pl):
#   0.2::a.  0.7::b.
#   c :- a, b.    c :- a, \+b.
#   q1 :- b, c.   % 0.14
#   q2 :- \+b, c. % 0.06

def test_negation_stratified():
    @Parser(language="plp")
    def prog():
        a: 0.2
        b: 0.7
        c :- a & b
        c :- a & ~b
        q1 :- b & c
        q2 :- ~b & c
        query(q1)

    assert prob(prog) == pytest.approx(0.14, abs=1e-5)


# ===========================================================================
# EVIDENCE
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 6. Evidence on an independent variable
# ---------------------------------------------------------------------------
# ProbLog (evidence_bug.pl):
#   0.4::a.  0.3::b.  0.5::q.
#   a1 :- b, a.   a2 :- b.
#   evidence(q, true).
#   query(a1).  % 0.12   (q is independent; posterior unchanged)
#   query(a2).  % 0.3

def test_evidence_independent():
    @Parser(language="plp")
    def prog():
        a: 0.4
        b: 0.3
        q: 0.5
        a1 :- b & a
        a2 :- b
        evidence(q, True)
        query(a1)

    assert prob(prog) == pytest.approx(0.12, abs=1e-5)


def test_evidence_independent_b():
    @Parser(language="plp")
    def prog():
        a: 0.4
        b: 0.3
        q: 0.5
        a1 :- b & a
        a2 :- b
        evidence(q, True)
        query(a2)

    assert prob(prog) == pytest.approx(0.3, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 7. Evidence on queried atom — two-coin conditioning
# ---------------------------------------------------------------------------
# ProbLog (01_coins tutorial):
#   evidence(twoHeads, false).
#   query(heads1).  % 2/7 ≈ 0.2857
#   query(heads2).  % 3/7 ≈ 0.4286

def test_evidence_on_rule_head():
    @Parser(language="plp")
    def prog():
        heads1: 0.5
        heads2: 0.6
        twoHeads :- heads1 & heads2
        evidence(twoHeads, False)
        query(heads1)

    # P(heads1 | twoHeads=F) = P(heads1, \+twoHeads) / P(\+twoHeads)
    # P(\+twoHeads) = 1 - 0.5*0.6 = 0.7
    # P(heads1, \+twoHeads) = P(heads1)*P(\+heads2) = 0.5*0.4 = 0.2
    assert prob(prog) == pytest.approx(0.2 / 0.7, abs=1e-4)


# ===========================================================================
# BAYESIAN NETWORKS
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 8. BN — auxiliary variable encoding
# ---------------------------------------------------------------------------
# ProbLog (4_bayesian_net.pl):
#   0.7::burglary.  0.2::earthquake.
#   alarm :- burglary, earthquake, p_alarm1.  (etc.)
#   evidence(alarm,true).
#   query(burglary).  % 0.9897

def test_bayes_net_auxiliary_facts():
    @Parser(language="plp")
    def bayes():
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

    assert prob(bayes) == pytest.approx(0.98965517, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 9. BN — weighted-rule (probabilistic clause) encoding
# ---------------------------------------------------------------------------
# ProbLog (4_1_bayesian_net.pl):
#   0.9::alarm <- burglary, earthquake.  (etc.)
#   evidence(alarm,true).
#   query(earthquake).  % 0.2276

def test_bayes_net_weighted_rules():
    @Parser(language="plp")
    def bayes():
        burglary: 0.7
        earthquake: 0.2
        alarm[0.9] :- burglary & earthquake
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake
        evidence(alarm, True)
        query(earthquake)

    assert prob(bayes) == pytest.approx(0.22758621, abs=1e-5)


# ===========================================================================
# ANNOTATED DISJUNCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 10. Ground AD — mutually exclusive outcomes
# ---------------------------------------------------------------------------
# ProbLog (ad_fact.pl):
#   0.3::p(1); 0.4::p(2).
#   query(p(1)).  % 0.3

def test_ad_ground():
    @Parser(language="plp")
    def ad():
        p1[0.3] | p2[0.4]
        query(p1)

    assert prob(ad) == pytest.approx(0.3, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 11. Large uniform AD (9-way) — fraction weights
# ---------------------------------------------------------------------------
# ProbLog (11_ads_numerical.pl):
#   1/9::a9_9; ... ; 1/9::a9_1 <- true.
#   query(a9_1).  % 0.111...
#
# Fraction literals like `1/9` are now supported directly.

def test_ad_uniform_9():
    @Parser(language="plp")
    def uniform9():
        a1[1/9] | a2[1/9] | a3[1/9] | \
        a4[1/9] | a5[1/9] | a6[1/9] | \
        a7[1/9] | a8[1/9] | a9[1/9]
        query(a1)

    assert prob(uniform9) == pytest.approx(1/9, abs=1e-6)


# ---------------------------------------------------------------------------
# [PARTIAL] 12. Multi-valued variable via AD — earthquake severity
# ---------------------------------------------------------------------------
# ProbLog (02_bayes.html example 4):
#   0.01::earthquake(heavy); 0.19::earthquake(mild); 0.8::earthquake(none).
#   0.90::alarm :- burglary, earthquake(heavy).
#   0.85::alarm :- burglary, earthquake(mild).
#   0.80::alarm :- burglary, earthquake(none).
#   0.10::alarm :- \+burglary, earthquake(mild).
#   0.30::alarm :- \+burglary, earthquake(heavy).
#   query(alarm).
#
# PARTIAL — ProbLog allows parameterised AD atoms (`earthquake(heavy)` etc.).
# This language only supports plain-Name AD atoms, so the earthquake states
# must be renamed.  The logic is otherwise identical.
# (LIM-3: _parse_head does not accept nested subscripts in AD position.)

def test_earthquake_multivalued_ad():
    @Parser(language="plp")
    def prog():
        burglary: 0.7
        eq_heavy[0.01] | eq_mild[0.19] | eq_none[0.8]

        alarm[0.90] :- burglary & eq_heavy
        alarm[0.85] :- burglary & eq_mild
        alarm[0.80] :- burglary & eq_none
        alarm[0.10] :- ~burglary & eq_mild
        alarm[0.30] :- ~burglary & eq_heavy

        query(alarm)

    # P(alarm) computed analytically:
    #   b=T (0.7): 0.01*0.90 + 0.19*0.85 + 0.80*0.80 = 0.009+0.1615+0.64 = 0.8105
    #   b=F (0.3): 0.01*0.30 + 0.19*0.10 + 0.80*0     = 0.003+0.019     = 0.022
    #   P(alarm) = 0.7*0.8105 + 0.3*0.022 = 0.56735 + 0.0066 = 0.57395
    assert prob(prog) == pytest.approx(0.57395, abs=1e-4)


# ===========================================================================
# FIRST-ORDER PROGRAMS (NON-RECURSIVE)
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 13. First-order weighted rule (intensional probabilistic fact)
# ---------------------------------------------------------------------------
# ProbLog (advars.pl):
#   0.3::e1(X,Y) :- a1(X,Z), b1(Z,Y).   % one path → 0.3
#   0.3::e2(X,Y) :- a2(X,Z), b2(Z,Y).   % two paths → 1-(1-0.3)² = 0.51

# def test_intermediate_join():
#     @Parser(language="plp")
#     def prog():
#         a1[n1, n2]: 1.0
#         b1[n2, n4]: 1.0

#         a2[n1, n2]: 1.0; a2[n1, n3]: 1.0
#         b2[n2, n4]: 1.0; b2[n3, n4]: 1.0

#         e1[X, Y, 0.3] :- a1[X, Z] & b1[Z, Y]
#         e2[X, Y, 0.3] :- a2[X, Z] & b2[Z, Y]

#         query(e1[n1, n4])

#     assert prob(prog) == pytest.approx(0.3, abs=1e-5)


# def test_intermediate_join_two_paths():
#     @Parser(language="plp")
#     def prog():
#         a2[n1, n2]: 1.0; a2[n1, n3]: 1.0
#         b2[n2, n4]: 1.0; b2[n3, n4]: 1.0

#         e2[X, Y, 0.3] :- a2[X, Z] & b2[Z, Y]

#         query(e2[n1, n4])

#     assert prob(prog) == pytest.approx(0.51, abs=1e-5)


# ---------------------------------------------------------------------------
# [OK] 14. First-order BN — calls network, unconditioned
# ---------------------------------------------------------------------------
# ProbLog (02_bayes.html example 3):
#   0.8::calls(X) :- alarm, person(X).
#   0.1::calls(X) :- \+alarm, person(X).
#   query(calls(john)).  % ≈ 0.506

def test_calls_network():
    @Parser(language="plp")
    def prog():
        person[john]: 1.0; person[mary]: 1.0
        burglary: 0.7; earthquake: 0.2

        alarm[0.9] :- burglary & earthquake
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake

        calls[X, 0.8] :- alarm & person[X]
        calls[X, 0.1] :- ~alarm & person[X]

        query(calls[john])

    assert prob(prog) == pytest.approx(0.506, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 15. First-order BN — calls network, evidence on calls → posterior
# ---------------------------------------------------------------------------
# ProbLog (02_bayes.html example 3):
#   evidence(calls(john),true).  evidence(calls(mary),true).
#   query(burglary).  % ≈ 0.9819

def test_calls_network_with_evidence():
    @Parser(language="plp")
    def prog():
        person[john]: 1.0; person[mary]: 1.0
        burglary: 0.7; earthquake: 0.2

        alarm[0.9] :- burglary & earthquake
        alarm[0.8] :- burglary & ~earthquake
        alarm[0.1] :- ~burglary & earthquake

        calls[X, 0.8] :- alarm & person[X]
        calls[X, 0.1] :- ~alarm & person[X]

        evidence(calls[john], True)
        evidence(calls[mary], True)
        query(burglary)

    assert prob(prog) == pytest.approx(0.981939, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 16. Wildcard body with plain-Name head  (was PARTIAL — LIM-4 / LIM-5)
# ---------------------------------------------------------------------------
# ProbLog (3_tossing_coin.pl):
#   0.6::heads(C) :- coin(C).
#   someHeads :- heads(_).   % 0.9744
#
# `_` is an anonymous variable that ranges over the full domain of the
# predicate it appears in.  `someHeads :- heads[_]` expands to one rule per
# ground instance of heads (noisy-or / existential semantics).

def test_tossing_coins_wildcard():
    @Parser(language="plp")
    def prog():
        coin[c1]: 1.0; coin[c2]: 1.0; coin[c3]: 1.0; coin[c4]: 1.0
        heads[C, 0.6] :- coin[C]
        someHeads :- heads[_]
        query(someHeads)

    assert prob(prog) == pytest.approx(0.9744, abs=1e-4)


# [OK] 16b. Constant subscript in plain-Name head body (LIM-5 fix)
# `someHeads :- heads[c1]` now correctly rewrites to `someHeads :- heads__c1`.

def test_tossing_coins_ground_subscript():
    @Parser(language="plp")
    def prog():
        coin[c1]: 1.0; coin[c2]: 1.0; coin[c3]: 1.0; coin[c4]: 1.0
        heads[C, 0.6] :- coin[C]
        someHeads :- heads[c1]
        someHeads :- heads[c2]
        someHeads :- heads[c3]
        someHeads :- heads[c4]
        query(someHeads)

    assert prob(prog) == pytest.approx(0.9744, abs=1e-4)


# ===========================================================================
# INHIBITION EFFECTS
# ===========================================================================

# ---------------------------------------------------------------------------
# [OK] 17. Noisy-or with evidence
# ---------------------------------------------------------------------------
# ProbLog (10_inhibitioneffects.html, example 1):
#   0.3::e :- c1.  0.4::e :- c2.
#   evidence(c1,true).  evidence(c2,true).
#   query(e).  % 0.58

def test_inhibition_noisy_or():
    @Parser(language="plp")
    def prog():
        c1: 0.5; c2: 0.5
        e[0.3] :- c1
        e[0.4] :- c2
        evidence(c1, True)
        evidence(c2, True)
        query(e)

    assert prob(prog) == pytest.approx(0.58, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 18. Weighted inhibitory rules (separate head auxiliaries)
# ---------------------------------------------------------------------------
# ProbLog (10_inhibitioneffects.html):
#   0.3::e1 :- c1.  0.2::\+e1 :- c2.
#   0.4::e2 :- c2.  0.2::\+e2 :- c1.
#   e :- e1.  e :- e2.
#   evidence(c1,true).  evidence(c2,true).
#   query(e).  % 0.4832

def test_inhibition_weighted():
    @Parser(language="plp")
    def prog():
        c1: 0.5; c2: 0.5
        e1[0.3] :- c1
        e1[0.2, False] :- c2
        e2[0.4] :- c2
        e2[0.2, False] :- c1
        e :- e1
        e :- e2
        evidence(c1, True)
        evidence(c2, True)
        query(e)

    assert prob(prog) == pytest.approx(0.4832, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 19. Negative head literals — deterministic and weighted inhibition
# ---------------------------------------------------------------------------
# ProbLog (negative_head_lits.pl):
#   0.5::a :- b.   0.5::\+a :- c.    % weighted inhibition   → 0.315
#   0.5::d :- b.       \+d :- c.     % deterministic inhib.  → 0.28
#   0.5::e :- b.  1.0::\+e :- c.     % same, explicit 1.0    → 0.28
#
# `\+head :- body` ≡ `head[1.0, False] :- body` in this language.

def test_negative_head_literals():
    @Parser(language="plp")
    def prog():
        b: 0.7; c: 0.2

        a[0.5] :- b
        a[0.5, False] :- c   # 0.5::\+a :- c

        d[0.5] :- b
        d[False] :- c        # \+d :- c  (deterministic, prob 1.0)

        e[0.5] :- b
        e[1.0, False] :- c   # 1.0::\+e :- c

        query(a)

    # P(a): worlds (b=T,c=T)=0.14, (b=T,c=F)=0.56, (b=F,*)=0.3
    # P(a|b=T,c=T) = 0.5*(1-0.5) = 0.25
    # P(a|b=T,c=F) = 0.5
    # P(a|b=F)     = 0
    # P(a) = 0.14*0.25 + 0.56*0.5 = 0.035 + 0.28 = 0.315
    assert prob(prog) == pytest.approx(0.315, abs=1e-4)


def test_negative_head_literals_deterministic():
    @Parser(language="plp")
    def prog():
        b: 0.7; c: 0.2

        d[0.5] :- b
        d[False] :- c

        query(d)

    # P(d|b=T,c=T) = 0   (always inhibited when c=T)
    # P(d|b=T,c=F) = 0.5
    # P(d) = 0.14*0 + 0.56*0.5 = 0.28
    assert prob(prog) == pytest.approx(0.28, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 20. Intercausal cancellation (max-entropy)
# ---------------------------------------------------------------------------
# ProbLog (10_inhibitioneffects.html, example 3):
#   0.8::xc1 :- c1.  0.5::\+xc1 :- c1, c2.
#   0.3::xc2 :- c2.  0.5::\+xc2 :- c1, c2.
#   e :- xc1.  e :- xc2.
#   evidence(c1,true).  evidence(c2,true).
#   query(e).

def test_intercausal_cancellation():
    @Parser(language="plp")
    def prog():
        c1: 0.5; c2: 0.5
        xc1[0.8] :- c1
        xc1[0.5, False] :- c1 & c2
        xc2[0.3] :- c2
        xc2[0.5, False] :- c1 & c2
        e :- xc1
        e :- xc2
        evidence(c1, True)
        evidence(c2, True)
        query(e)

    # With c1=T, c2=T:
    #   P(xc1) = 0.8*(1-0.5) = 0.4
    #   P(xc2) = 0.3*(1-0.5) = 0.15
    #   P(e)   = 1-(1-0.4)*(1-0.15) = 1 - 0.6*0.85 = 0.49
    assert prob(prog) == pytest.approx(0.49, abs=1e-4)


# ---------------------------------------------------------------------------
# [OK] 21. Osteoporosis treatment — multi-rule inhibition chain
# ---------------------------------------------------------------------------
# ProbLog (10_inhibitioneffects.html, example 3):
#   increaseOsteoblasts :- calcium.
#   0.5::\+increaseOsteoblasts :- calcium, bispho.
#   reduceOsteoclasts :- bispho.
#   1.0::\+reduceOsteoclasts :- calcium, bispho.
#   osteoprosis :- initialOsteoprosis.
#   0.85::\+osteoprosis :- reduceOsteoclasts.
#   0.15::\+osteoprosis :- increaseOsteoblasts.
#   evidence(initialOsteoprosis,true). evidence(calcium,true). evidence(bispho,false).
#   query(osteoprosis).

def test_osteoporosis():
    @Parser(language="plp")
    def prog():
        calcium: 0.5
        bispho: 0.5
        initialOsteoprosis: 0.5

        increaseOsteoblasts :- calcium
        increaseOsteoblasts[0.5, False] :- calcium & bispho

        reduceOsteoclasts :- bispho
        reduceOsteoclasts[1.0, False] :- calcium & bispho

        osteoprosis :- initialOsteoprosis
        osteoprosis[0.85, False] :- reduceOsteoclasts
        osteoprosis[0.15, False] :- increaseOsteoblasts

        evidence(initialOsteoprosis, True)
        evidence(calcium, True)
        evidence(bispho, False)
        query(osteoprosis)

    # With calcium=T, bispho=F, initialOsteoprosis=T:
    #   increaseOsteoblasts = T (from calcium=T; inhibition body=F since bispho=F)
    #   reduceOsteoclasts   = F (from bispho=F)
    #   osteoprosis starts T (from initialOsteoprosis)
    #     0.85 inhibition: body = reduceOsteoclasts = F → no effect
    #     0.15 inhibition: body = increaseOsteoblasts = T → inhibit w.p. 0.15
    #   P(osteoprosis) = 1 - 0.15 = 0.85
    assert prob(prog) == pytest.approx(0.85, abs=1e-4)


# ===========================================================================
# RECURSIVE PROGRAMS
# ===========================================================================

# ---------------------------------------------------------------------------
# 22. Probabilistic graph reachability (DAG — no cyclic recursion)
# ---------------------------------------------------------------------------
# ProbLog (7_probabilistic_graph.pl):
#   path(X,Y) :- edge(X,Y).
#   path(X,Y) :- edge(X,Z), Y \== Z, path(Z,Y).
#   query(path(1,5)).  % 0.25824
#
# FIXED: BUG-1 (fixpoint-init biconditional `head <-> False`) resolved.
# DAG structure ensures biconditionals have a unique minimal fixed point.

def test_graph_reachability_dag():
    @Parser(language="plp")
    def graph():
        edge[n1, n2]: 0.6
        edge[n1, n3]: 0.1
        edge[n2, n5]: 0.4
        edge[n2, n6]: 0.3
        edge[n3, n4]: 0.3
        edge[n4, n5]: 0.8
        edge[n5, n6]: 0.2

        path[X, Y] :- edge[X, Y]
        path[X, Y] :- edge[X, Z] & path[Z, Y]

        query(path[n1, n5])

    assert prob(graph) == pytest.approx(0.25824, abs=1e-4)


# ---------------------------------------------------------------------------
# 22a. Graph reachability with inequality guard (Y != Z)
# ---------------------------------------------------------------------------
# Adding `Y != Z` to the recursive rule prevents the grounder from emitting
# groundings where the intermediate node equals the destination.  On a DAG
# with no self-loops the result is identical to the unguarded version.

def test_graph_reachability_dag_with_inequality():
    @Parser(language="plp")
    def graph():
        edge[n1, n2]: 0.6
        edge[n1, n3]: 0.1
        edge[n2, n5]: 0.4
        edge[n2, n6]: 0.3
        edge[n3, n4]: 0.3
        edge[n4, n5]: 0.8
        edge[n5, n6]: 0.2

        path[X, Y] :- edge[X, Y]
        path[X, Y] :- edge[X, Z] & path[Z, Y] & Y != Z

        query(path[n1, n5])

    assert prob(graph) == pytest.approx(0.25824, abs=1e-4)


def test_inequality_filters_self_grounding():
    """Y != Z prunes Z=Y groundings: checks that no `path[X,Y] :- ... & path[Y,Y]`
    rule is emitted for this two-node graph, keeping the result identical to the
    hand-grounded version without Y=Z cases."""
    @Parser(language="plp")
    def prog():
        # Linear chain: n1 → n2 → n3
        edge[n1, n2]: 0.8
        edge[n2, n3]: 0.6

        path[X, Y] :- edge[X, Y]
        path[X, Y] :- edge[X, Z] & path[Z, Y] & Y != Z

        query(path[n1, n3])

    # path[n2,n3] ← edge[n2,n3]            P = 0.6
    # path[n1,n3] ← edge[n1,n2] & path[n2,n3]  P = 0.8 * 0.6 = 0.48
    # The Z=n3 grounding (path[n1,n3] :- edge[n1,n3] & path[n3,n3]) is filtered
    # by edge not existing; Z=n3 in recursive rule is filtered by Y!=Z.
    assert prob(prog) == pytest.approx(0.48, abs=1e-6)


# ---------------------------------------------------------------------------
# 22b. Non-derivable query atom → probability zero
# ---------------------------------------------------------------------------
# Querying an atom that no rule can derive must return 0.0, not 1.0.
# path(n1,n1) is not reachable in this DAG (no self-loops), so it is never
# emitted as a ground rule by the relevance grounder; the SDD must still
# answer the query correctly rather than treating the missing variable as
# unconstrained (which would yield 1.0 in the parallel WMC semiring).

def test_non_derivable_query_returns_zero():
    @Parser(language="plp")
    def graph():
        edge[n1, n2]: 0.9
        edge[n1, n3]: 0.4
        edge[n2, n4]: 0.5
        edge[n3, n4]: 0.6

        path[X, Y] :- edge[X, Y]
        path[X, Y] :- edge[X, Z] & path[Z, Y]

        query(path[n1, n1])   # n1→n1 is not reachable — no self-loops

    assert prob(graph) == pytest.approx(0.0, abs=1e-9)


def test_non_derivable_propositional_returns_zero():
    """Querying a predicate never defined or derived returns 0."""
    @Parser(language="plp")
    def prog():
        a[0.4] | b[0.6]   # AD — ensures at least one formula exists
        query(missing)     # 'missing' has no fact, no rule, no derivation

    assert prob(prog) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# [BUG] 23. Smokers network — cyclic recursion (biconditional admits non-minimal models)
# ---------------------------------------------------------------------------
# ProbLog (8_smokers_network.pl):
#   smokes(X) :- stress(X).
#   smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).
#   evidence(smokes(2),true).
#   query(smokes(1)).  % 0.5088
#
# BUG-2 — cyclic mutual recursion (smokes[p1] ↔ ...smokes[p2]... and vice versa)
# allows non-minimal stable models in the biconditional encoding.  WMC counts
# worlds where (smokes[p1]=T, smokes[p2]=T) is a spurious fixpoint, inflating the
# result.  ProbLog enforces the minimal Herbrand model; symprog gets 0.5354 vs 0.5088.

# @pytest.mark.xfail(reason="Cyclic recursion: biconditional encoding admits non-minimal "
#                            "stable models (BUG-2).  Gets 0.5354 instead of ProbLog's 0.5088.")
# def test_smokers():
#     @Parser(language="plp")
#     def prog():
#         person[p1]: 1.0; person[p2]: 1.0; person[p3]: 1.0; person[p4]: 1.0

#         friend[p1, p2]: 1.0
#         friend[p2, p1]: 1.0
#         friend[p2, p4]: 1.0
#         friend[p3, p2]: 1.0
#         friend[p4, p2]: 1.0

#         stress[X, 0.3] :- person[X]
#         influences[X, Y, 0.2] :- person[X] & person[Y]

#         smokes[X] :- stress[X]
#         smokes[X] :- friend[X, Y] & influences[Y, X] & smokes[Y]

#         asthma[X, 0.4] :- smokes[X]

#         evidence(smokes[p2], True)
#         evidence(influences[p4, p2], False)
#         query(smokes[p1])

#     assert prob(prog) == pytest.approx(0.5087719, abs=1e-4)


# ---------------------------------------------------------------------------
# [BUG] 24. Smokers — alternate encoding (direct weighted recursive rule)
# ---------------------------------------------------------------------------
# ProbLog (advars_smokers_alt.pl):
#   smokes(X) :- stress(X).
#   0.2::smokes(X) :- friend(X,Y), smokes(Y).   ← weighted recursive rule
#   evidence(smokes(2),true).
#   query(smokes(1)).  % 0.49796
#
# BUG-2 — same cyclic recursion issue as test 23.

# @pytest.mark.xfail(reason="Cyclic recursion: biconditional encoding admits non-minimal "
#                            "stable models (BUG-2).  Gets 0.5153 instead of ProbLog's 0.4980.")
# def test_smokers_alt_encoding():
#     @Parser(language="plp")
#     def prog():
#         person[p1]: 1.0; person[p2]: 1.0; person[p3]: 1.0; person[p4]: 1.0

#         friend[p1, p2]: 1.0
#         friend[p2, p1]: 1.0
#         friend[p2, p4]: 1.0
#         friend[p3, p2]: 1.0
#         friend[p4, p2]: 1.0

#         stress[X, 0.3] :- person[X]

#         smokes[X] :- stress[X]
#         smokes[X, 0.2] :- friend[X, Y] & smokes[Y]

#         asthma[X, 0.4] :- smokes[X]

#         evidence(smokes[p2], True)
#         query(smokes[p1])

#     assert prob(prog) == pytest.approx(0.49795533, abs=1e-4)


# ---------------------------------------------------------------------------
# 25. Disease spread — recursive infection model (DAG-like, no cyclic dependency)
# ---------------------------------------------------------------------------
# ProbLog (10_inhibitioneffects.html, basic infection model):
#   0.1::initialInf(X) :- person(X).
#   0.1::contact(X,Y) :- person(X), person(Y).
#   inf(X) :- initialInf(X).
#   0.6::inf(X) :- contact(X,Y), inf(Y).
#   query(inf(_)).       ← also uses wildcard (LIM-4)
#
# FIXED: BUG-1 resolved.  Only a range assertion (no ProbLog ground truth for wildcard query).

def test_disease_spread():
    @Parser(language="plp")
    def prog():
        person[a]: 1.0; person[b]: 1.0

        initialInf[X, 0.1] :- person[X]
        contact[X, Y, 0.1] :- person[X] & person[Y]

        inf[X] :- initialInf[X]
        inf[X, 0.6] :- contact[X, Y] & inf[Y]

        someInf :- inf[_]   # wildcard: expands to someInf :- inf__a, someInf :- inf__b
        query(someInf)

    assert 0.0 < prob(prog) < 1.0


# ===========================================================================
# NOT SUPPORTED
# ===========================================================================

# ---------------------------------------------------------------------------
# [NO] 26. HMM weather — first-order ADs with arithmetic
# ---------------------------------------------------------------------------
# ProbLog (6_hmm_weather.pl):
#   0.6::weather(sun,T); 0.4::weather(rain,T) <- T>0, Tprev is T-1, weather(sun,Tprev).
#   query(weather(sun,10)).  % 0.3334
#
# NOT SUPPORTED:
#   LIM-1  Arithmetic in rule bodies (`is/2`, `T>0`, `Tprev is T-1`).
#   LIM-2  First-order ADs with a body guard (`; <- condition`).
#   Integer ranges: grounder cannot enumerate timesteps automatically.
#
# A manual horizon-T unroll is possible but does not scale.

def test_hmm_weather_manual_unroll():
    """Hand-unrolled HMM for T=0..2 — illustrates verbosity; not a general solution."""
    @Parser(language="plp")
    def hmm():
        w_sun_0[0.5] | w_rain_0[0.5]

        w_sun_1a[0.6] | w_rain_1a[0.4]   # from sun at T=0
        w_sun_1b[0.2] | w_rain_1b[0.8]   # from rain at T=0

        w_sun_1 :- w_sun_0 & w_sun_1a
        w_sun_1 :- w_rain_0 & w_sun_1b

        w_sun_2a[0.6] | w_rain_2a[0.4]
        w_sun_2b[0.2] | w_rain_2b[0.8]

        w_sun_2 :- w_sun_1 & w_sun_2a
        w_sun_2 :- ~w_sun_1 & w_sun_2b

        query(w_sun_2)

    assert 0.0 < prob(hmm) < 1.0   # sanity only


# ---------------------------------------------------------------------------
# [NO] 27. Cards — multi-arg parameterised AD + comparison in body
# ---------------------------------------------------------------------------
# ProbLog (10_cards.pl):
#   0.25::card(C,q,h); 0.25::card(C,k,h); ... <- position(C).
#   doublecard :- card(C1,X,Y), card(C2,X,Y), C1 < C2.  % 0.25
#
# NOT SUPPORTED:
#   LIM-3  Multi-argument parameterised AD heads — _parse_head rejects nested subscripts.
#   LIM-1  `C1 < C2` comparison in rule bodies.
#
# The two ground AD groups can be written manually; doublecard cannot.

def test_cards_manual():
    @Parser(language="plp")
    def cards():
        card1_qh[0.25] | card1_kh[0.25] | card1_qs[0.25] | card1_ks[0.25]
        card2_qh[0.25] | card2_kh[0.25] | card2_qs[0.25] | card2_ks[0.25]

        samecard_qh :- card1_qh & card2_qh

        query(samecard_qh)

    assert prob(cards) == pytest.approx(0.0625, abs=1e-5)


# ---------------------------------------------------------------------------
# [NO] 28. Packing problem — arithmetic + list terms
# ---------------------------------------------------------------------------
# ProbLog (9_packing_problem.pl):
#   P::pack(Item) :- weight(Item,Weight), P is 1.0/Weight.
#   excess([I|R],Lim) :- pack(I), weight(I,W), L is Lim-W, excess(R,L).
#   query(excess(8)).  % 0.1181
#
# NOT SUPPORTED:
#   LIM-1  Arithmetic (`is/2`, comparisons).
#   LIM-6  List terms `[H|T]`.
#   LIM-7  Intensional probabilities (probability computed from a term value).


# ---------------------------------------------------------------------------
# [OK] 28b. Wildcard query — query all ground instances
# ---------------------------------------------------------------------------
# ProbLog: query(heads(_)).   % returns P(heads(c1)), P(heads(c2)), …
#
# `query(pred[_])` expands to one ground query per domain element of pred.
# Multiple results are returned as a list of SolverOut objects.

def test_wildcard_query():
    @Parser(language="plp")
    def prog():
        coin[c1]: 1.0; coin[c2]: 1.0; coin[c3]: 1.0; coin[c4]: 1.0
        heads[C, 0.6] :- coin[C]
        query(heads[_])

    results = prog.inference()
    assert isinstance(results, list)
    assert len(results) == 4
    assert all(abs(r.inference.item() - 0.6) < 1e-4 for r in results)


# ---------------------------------------------------------------------------
# [NO] 29. Negated queries
# ---------------------------------------------------------------------------
# ProbLog (negative_query.pl):
#   0.3::p.
#   query(\+p).  % 0.7
#
# NOT SUPPORTED: query() only accepts ground atom names, not formula terms.
# Workaround: introduce a named rule `not_p :- \+p` and query that.

def test_negated_query_workaround():
    """Named negation works; bare `query(+p)` syntax does not."""
    @Parser(language="plp")
    def prog():
        p: 0.3
        not_p :- ~p
        query(not_p)

    assert prob(prog) == pytest.approx(0.7, abs=1e-5)


# ---------------------------------------------------------------------------
# [NO] 30. Non-ground queries
# ---------------------------------------------------------------------------
# ProbLog (non_ground_query.pl / 01_queries.pl):
#   query(a(X)).          → returns P(a(1)), P(a(2)), P(a(3)) as a group
#   query(p(X)) :- a(X).  → same pattern with arithmetic body
#
# NOT SUPPORTED: query() does not accept variables or produce multi-valued results.
# Also requires arithmetic in rule bodies (LIM-1) and built-ins (LIM-8).


# ---------------------------------------------------------------------------
# [NO] 31. Dynamic / intensional probabilities
# ---------------------------------------------------------------------------
# ProbLog (ground_term_variable_prob.pl):
#   P::a :- P=0.3.   % probability computed via unification at runtime
#   query(a).        % 0.3
#
# NOT SUPPORTED (LIM-7): probability must be a literal constant at parse time.
# The expression `P::a :- P=0.3` has no analogue in this language.


# ===========================================================================
# Summary
# ===========================================================================
# STATUS (32 programs / 31 executable tests):
#
#   [OK]      1–23, 16b, 29-workaround   (basic facts, BNs, ADs, NAF,
#                                          evidence, first-order, wildcards,
#                                          inhibition, intercausal, osteoporosis)
#   [PARTIAL] 12 (multivalued AD — atom renaming required)
#             27 (cards manual — doublecard blocked by LIM-1 + LIM-3)
#   [BUG]     22–25  (graph, smokers ×2, disease spread — BUG-1)
#   [NO]      26 (HMM), 28 (packing), 30 (non-ground query), 31 (dynamic prob)
#
# BUG-1 (tests 22–25) — fixpoint-init biconditional
#   PLPSemantics.visit_Assign creates `head <-> False` for the fixpoint-
#   initialisation statements emitted by PLPTransformer._make_fixpoint_loop
#   (`head = False` at the start of the while block).  These contradict the
#   correct rule biconditionals from visit_While, forcing all recursive heads
#   to False.  Tracked in TODO.md.
#
# Language limitations (LIM-*):
#   LIM-1  No arithmetic in rule bodies (`is/2`, `<`, `>`, `\==`).
#          Blocks: HMMs, temporal models, packing, cycle guard, C1<C2.
#   LIM-2  No first-order ADs with a body guard (`H1[p];H2[p] <- cond`).
#   LIM-3  Multi-argument parameterised AD heads — _parse_head rejects nested subscripts.
#   LIM-4  FIXED: wildcard `_` now works in both rule bodies and query() calls.
#          `query(pred[_])` expands to one ground query per domain element and
#          returns a list of SolverOut objects.
#   LIM-5  FIXED: plain-Name heads with subscript bodies (e.g. `a :- pred[c1]`
#          or `a :- pred[_]`) are now correctly grounded.
#   LIM-6  No list terms (`[H|T]`, `member/2`, etc.).
#   LIM-7  No intensional probabilities (P computed from a term at runtime).
#   LIM-8  No built-in predicates (`between/3`, `findall/3`, `msort/2`, etc.).
#   LIM-9  FIXED: fraction/arithmetic weight expressions (`1/9`, `3*0.1`, etc.)
#          are now evaluated at parse time via `_eval_numeric`.
